import torch 
import torch.nn as nn 
import torch.nn.functional as F

class SimpleBlock(nn.Module):
    def __init__(self, c_in, c_out, height, width) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([c_out, height, width]),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, height, width) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([channels, height, width]),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([channels, height, width]),
        )
        
    def forward(self, x):
        out = self.layer(x)
        out = out + x
        out = nn.functional.relu(out)
        return out 
    

class ForwardBlock(nn.Module):
    def __init__(self, c_in, c_out, height, width, resblocks=1) -> None:
        super().__init__()

        self.simple_block = SimpleBlock(c_in, c_out, height, width)
        self.res_blocks = nn.Sequential(*[ResidualBlock(c_out, height, width) for _ in range(resblocks)])

    def forward(self, x):
        x = self.simple_block(x)
        x = self.res_blocks(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, height, width, resblocks=1) -> None:
        super().__init__()
        self.forward_block = ForwardBlock(c_in, c_out, height, width, resblocks)
        self.max_pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.max_pool2d(x)
        x = self.forward_block(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, height, width, resblocks=1) -> None:
        super().__init__()
        self.forward_block = ForwardBlock(c_in, c_out, height, width, resblocks)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True)

    def forward(self, x, skip):
        x = self.up_sample(x)
        x = torch.cat((x,skip), dim = 1)
        x = self.forward_block(x)
        return x


class UnetFeatures(nn.Module):
    def __init__(self, height=512, width=512) -> None:
        super().__init__()

        self.d0 = ForwardBlock(3 ,64, height, width, 1) # c_in, c_out, height_out, width_out, resblocks
        self.d1 = DownBlock(64, 128, height//2, width//2, 1) # c_in, c_out, resblocks
        self.d2 = DownBlock(128, 256, height//4, width//4, 1)
        self.d3 = DownBlock(256, 512, height//8, width//8, 1)
        self.d4 = DownBlock(512, 1024, height//16, width//16, 1)
        self.d5 = DownBlock(1024, 1024, height//32, width//32, 2) #16

        self.u4 = UpBlock(1024+1024, 1024, height//16, width//16, 1)
        self.u3 = UpBlock(1024+512, 512, height//8, width//8, 1)

    
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
    def __init__(self, height=64, width=64) -> None:
        super().__init__()

        self.d0 = ForwardBlock(4096, 4096//2, height, width, 1) #H,W = 64,64
        self.d1 = DownBlock(4096//2, 4096//4, height//2, width//2, 1) #H,W = 32,32
        self.d2 = DownBlock(4096//4, 4096//8, height//4, width//4, 2) #H,W = 16,16

        self.u1 = UpBlock(4096//8 + 4096//4, 4096//8, height//2, width//2, 1)  #H,W = 32,32
        self.u0 = UpBlock(4096//8 + 4096//2, 4096//16, height, width, 1) #H,W = 64,64

        self.out = nn.Conv2d(4096//16, 4, kernel_size=3, padding=1)

    def forward(self, x):
        d0 = self.d0(x)
        d1 = self.d1(d0)
        d2 = self.d2(d1)

        u1 = self.u1(d2, d1)
        u0 = self.u0(u1, d0)
        out = self.out(u0)
        return out







