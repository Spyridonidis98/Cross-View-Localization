import torch 
import torch.nn as nn 
import torch.nn.functional as F

class SimpleBlock(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
    def forward(self, x):
        out = self.layer(x)
        out = out + x
        out = nn.functional.relu(out)
        return out 
    

class ForwardBlock(nn.Module):
    def __init__(self, c_in, c_out, resblocks=1) -> None:
        super().__init__()

        self.simple_block = SimpleBlock(c_in, c_out)
        self.res_blocks = nn.Sequential(*[ResidualBlock(c_out) for _ in range(resblocks)])

    def forward(self, x):
        x = self.simple_block(x)
        x = self.res_blocks(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, resblocks=1) -> None:
        super().__init__()
        self.forward_block = ForwardBlock(c_in, c_out, resblocks)
        self.max_pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.max_pool2d(x)
        x = self.forward_block(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, resblocks=1) -> None:
        super().__init__()
        self.forward_block = ForwardBlock(c_in, c_out, resblocks)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True)

    def forward(self, x, skip):
        x = self.up_sample(x)
        x = torch.cat((x,skip), dim = 1)
        x = self.forward_block(x)
        return x


class UnetFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.d0 = ForwardBlock(3 ,64, 1) # c_in, c_out, resblocks
        self.d1 = DownBlock(64, 128, 1) # c_in, c_out, resblocks
        self.d2 = DownBlock(128, 256, 1)
        self.d3 = DownBlock(256, 512, 1)
        self.d4 = DownBlock(512, 1024, 1)
        self.d5 = DownBlock(1024, 1024, 2) #16

        self.u4 = UpBlock(1024+1024, 1024, 1)
        self.u3 = UpBlock(1024+512, 512, 1)

    
    def forward(self, x):
        d0 = self.d0(x) #512
        d1 = self.d1(d0) #256
        d2 = self.d2(d1) #128
        d3 = self.d3(d2) #64
        d4 = self.d4(d3) #32
        d5 = self.d5(d4) #16

        u4 = self.u4(d5, d4) #32
        u3 = self.u3(u4, d3) #B, 512, 64, 64

        return u3

class UnetFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.d0 = ForwardBlock(4096, 4096//2, 1) #64
        self.d1 = DownBlock(4096//2, 4096//4, 1) #32
        self.d2 = DownBlock(4096//4, 4096//8, 2) #16

        self.u1 = UpBlock(4096//8 + 4096//4, 4096//8, 1) #32
        self.u0 = UpBlock(4096//8 + 4096//2, 4096//16, 1) #64

        self.out = nn.Conv2d(4096//16, 4, kernel_size=3, padding=1)

    def forward(self, x):
        d0 = self.d0(x)
        d1 = self.d1(d0)
        d2 = self.d2(d1)

        u1 = self.u1(d2, d1)
        u0 = self.u0(u1, d0)
        out = self.out(u0)
        return out

class Image2Token(nn.Module):
    def __init__(self, c_in, c_out, hw):
        super().__init__()

        self.proj = nn.Linear(c_in, c_out, bias=False)
        self.ln = nn.LayerNorm(c_out)
        self.pos_emb = nn.Embedding(hw, c_out)
        self.hw = hw

    def forward(self, x):
        B,C,H,W = x.shape

        x = x.view(B,C,H*W).transpose(1, 2)
        x = self.proj(x) + self.pos_emb((torch.arange(0, self.hw, dtype=torch.long, device=x.device)))
        x = self.ln(x)
        return x

class Token2Image(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        B,_,_ = x.shape #B,H*W,C
        x = x.transpose(1,2).view(B, self.C, self.H, self.W)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, n_emb, n_heads):
        '''self attention with skip connection inside'''
        super().__init__()
        assert n_emb%n_heads == 0

        self.n_emb = n_emb
        self.n_heads = n_heads

        self.c_atten = nn.Linear(n_emb, n_emb*3, bias=False)
        self.c_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.ln = nn.LayerNorm(n_emb)

    def forward(self, x):
        B,T,C = x.shape

        q, k, v = self.c_atten(x).split(self.n_emb, dim=-1)  #(B, T, 3*C) -> 3*(B, T, C)
        k = k.view(B, T, self.n_heads, C //  self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C //  self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C //  self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
   
        return self.ln(out)+x

class CrossAttention(nn.Module):
    def __init__(self, n_emb, n_heads):
        '''cross attention with skip connection inside'''
        super().__init__()
        assert n_emb%n_heads == 0

        self.n_emb = n_emb
        self.n_heads = n_heads

        self.atten_q = nn.Linear(n_emb, n_emb, bias=False)
        self.atten_kv = nn.Linear(n_emb, n_emb*2, bias=False)
        self.c_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.ln = nn.LayerNorm(n_emb)

    def forward(self, x, y):
        B, T, C = x.shape

        q = self.atten_q(x)
        k, v = self.atten_kv(y).split(self.n_emb, dim = -1) #(B, T, 2*C) - > 2*(B, T, C)

        q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2) #(B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2) #(B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2) #(B, nh, T, hs)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v) # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)

        return self.ln(out)+x


class MLP(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        '''MLP with skip connection inside'''
        self.ffw = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb, bias=True),
                nn.GELU(),
                nn.Linear(4 * n_emb, n_emb, bias=False),
                )
        
        self.ln = nn.LayerNorm(n_emb)

    def forward(self, x):
        out = self.ffw(x)
        return self.ln(out)+x

class Block(nn.Module):
    def __init__(self, n_emb, n_heads):
        super().__init__()

        self.self_att = SelfAttention(n_emb, n_heads)
        self.cross_att = CrossAttention(n_emb, n_heads)
        self.mlp = MLP(n_emb)

    def forward(self, x, y):
        x = self.cross_att(x, y)
        x = self.self_att(x)
        x = self.mlp(x)

        return x








