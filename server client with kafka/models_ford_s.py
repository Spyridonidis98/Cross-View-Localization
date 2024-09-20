import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import utils
import os
import torchvision.transforms.functional as TF

from VGG import VGGUnet, Encoder, Decoder2, Decoder4, VGGUnetTwoDec, Decoder, Unet
from jacobian import grid_sample

# from models_kitti import normalize_feature
# from transformer import LocalFeatureTransformer
# from position_encoding import PositionEncoding, PositionEncodingSine
from RNNs import NNrefine, Uncertainty
from swin_transformer import TransOptimizerS2GP_V1, TransOptimizerG2SP_V1
from swin_transformer_cross import TransOptimizerG2SP, TransOptimizerG2SPV2
from cross_attention import CrossViewAttention
import torch.jit as jit 

EPS = utils.EPS



class ModelFord(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(ModelFord, self).__init__()
        '''
        loss_method: 0: direct R T loss 1: feat loss 2: noise aware feat loss
        '''
        self.args = args

        self.level = args.level
        self.N_iters = args.N_iters


        self.SatFeatureNet = VGGUnet(self.level)

        if self.args.lifting != 'homography':
            self.unet0 = Unet(256)
            self.unet1 = Unet(128)
            self.unet2 = Unet(64)
            self.decoder_3dto2d_nets = [self.unet0, self.unet1, self.unet2]

        if self.args.proj == 'CrossAttn':
            self.GrdEnc = Encoder()
            self.GrdDec = Decoder()
            self.Dec4 = Decoder4()
            self.Dec2 = Decoder2()
            self.CVattn = CrossViewAttention(blocks=2, dim=256, heads=4, dim_head=16, qkv_bias=False)

        else:
            self.GrdFeatureNet = VGGUnet(self.level)

        self.TransRefine = TransOptimizerG2SP_V1()

        if self.args.rotation_range > 0:
            self.coe_R = nn.Parameter(torch.tensor(-5., dtype=torch.float32), requires_grad=True)
            self.coe_T = nn.Parameter(torch.tensor(-3., dtype=torch.float32), requires_grad=True)

        if self.args.use_uncertainty:
            self.uncertain_net = Uncertainty()

        # ori_grdH = 256
        # ori_grdW = 1024 
        self.xyz = self.get_xyz(self.args.lifting)

        self.Ks_list = {}
        for camera in args.cameras:
            K_list = [] 
            for level in range(4):
                K = args.Ks[camera].clone()
                K[0] = K[0] / (2 ** (3 - level))
                K[1] = K[1] / (2 ** (3 - level))
                K_list.append(K)

            self.Ks_list[camera] = K_list  # [1, 3, 3]


        meter_per_pixel = 0.22  # this is fixed for the ford dataset
        self.meters_per_pixel = []
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))


        self.corr_maps = None
        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.


    def get_xyz(self, mode = 'homography'):
        '''
        if mode homography returns list( [1, A/8, A/8, 3], [1, A/4, A/4, 3], , [1, A/2, A/2, 3])
        else returns list( [1, A/8, A/8, Z, 3], [1, A/4, A/4, Z, 3], , [1, A/2, A/2, Z, 3])
        '''
        ori_A = 512
        xyz = []
        self.ori_A = ori_A
        if mode == 'homography':
            for level in range(4):
                A = ori_A / (2 ** (3 - level))
                xyz.append(self.sat2world(A)) # returns [1, sat_map_H, sat_map_H, 3] under the grd camera coordinates sat_map_H = sat_map_W
        else:
            for level in range(4):
                z = []
                for z_offset in (torch.arange(8, dtype=torch.float32)/7 - 1) * 3.2: 
                    A = ori_A / (2 ** (3 - level))
                    zi = self.sat2world(A, z_offset)
                    zi = zi.unsqueeze(3)
                    z.append(zi)
                xyz.append(torch.cat(z, dim = 3))
        return xyz

    def sat2world(self, satmap_sidelength, z_offset = 0):
        # realword: X: North, Y:East, Z: Down   origin is set to the height of camera
        # the world is relative to axis of vehicle body 

        # the axis of the car body is positioned at the center of rear axel of the car which is above the ground so we have to take this distnace into account
        ground2carbody = 0.3 #given that the avarage height of a vehicles tire is 0.6 meters 


        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor([u0, v0]).cuda()

        meter_per_pixel = 0.22 # this is fixed for the ford dataset
        meter_per_pixel *= self.ori_A / satmap_sidelength
        R = torch.tensor([[0, -1], [1, 0]]).float().cuda()
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        XY = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Z = torch.ones_like(XY[..., :1])*ground2carbody + z_offset

        XYZ = torch.cat([XY, Z], dim=-1).unsqueeze(dim=0)  # [1, satmap_sidelength, satmap_sidelength, 3]

        return XYZ

    def World2GrdImgPixCoordinates(self, R_FL, T_FL, shift_u, shift_v, theta, level, camera):
        B = shift_u.shape[0]
        if len(self.xyz[level].shape) == 4:
            Xw = self.xyz[level].detach().to(shift_u.device).repeat(B, 1, 1, 1) # [B, X, Y, 3]
        
        if len(self.xyz[level].shape) == 5:
            Xw = self.xyz[level].detach().to(shift_u.device).repeat(B, 1, 1, 1, 1) # [B, X, Y, Z, 3]

        
        shift_u_meters = self.args.shift_range_lat * shift_u
        shift_v_meters = self.args.shift_range_lon * shift_v

        Tw = torch.cat([-shift_v_meters, shift_u_meters, torch.zeros_like(shift_v_meters)], dim=-1)  # [B, 3]

        yaw = theta * self.args.rotation_range / 180 * np.pi
        cos = torch.cos(yaw)
        sin = torch.sin(yaw)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        Rw = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B, 9]
        Rw = Rw.view(B, 3, 3)  # shape = [B, 3, 3]

        if len(self.xyz[level].shape) == 4:
            Xb = torch.sum(Rw[:, None, None, :, :] * Xw[:, :, :, None, :], dim=-1) + Tw[:, None, None, :]
        if len(self.xyz[level].shape) == 5:       
            Xb = torch.sum(Rw[:, None, None, None, :, :] * Xw[:, :, :, :, None, :], dim=-1) + Tw[:, None, None, None, :]


        K_FL = self.Ks_list[camera][level].detach().to(shift_u.device).repeat(B, 1, 1)
        R_FL = R_FL[camera].detach().to(shift_u.device).unsqueeze(0).repeat(B, 1, 1)
        T_FL = T_FL[camera].detach().to(shift_u.device).unsqueeze(0).repeat(B, 1)
        R_FL_inv = torch.inverse(R_FL)
        KR_FL = torch.matmul(K_FL, R_FL_inv)

        if len(self.xyz[level].shape) == 4:
            uvw = torch.sum(KR_FL[:, None, None, :, :] * (Xb[:, :, :, None, :] - T_FL[:, None, None, None, :]), dim=-1)
        if len(self.xyz[level].shape) == 5: 
            uvw = torch.sum(KR_FL[:, None, None, None, :, :] * (Xb[:, :, :, :, None, :] - T_FL[:, None, None, None, None, :]), dim=-1)
        # [B, H, W, 3]

        # Xm = Xb[:, :, :, None, :] - T_FL[:, None, None, None, :]

        if len(self.xyz[level].shape) == 4:
            denominator = torch.maximum(uvw[:, :, :, 2:], torch.ones_like(uvw[:, :, :, 2:]) * 1e-6)
        if len(self.xyz[level].shape) == 5:
            denominator = torch.maximum(uvw[:, :, :, :, 2:], torch.ones_like(uvw[:, :, :, :,2:]) * 1e-6)
        uv = uvw[..., :2] / denominator

        # H, W = uv.shape[1:-1]
        # assert(H==W)

        if len(self.xyz[level].shape) == 4:
            mask = torch.greater(denominator, torch.ones_like(uvw[:, :, :, 2:]) * 1e-6)
        if len(self.xyz[level].shape) == 5:
            mask = torch.greater(denominator, torch.ones_like(uvw[:, :, :, :, 2:]) * 1e-6)

        uv = uv * mask

        return uv, mask[..., 0]

    def project_grd_to_sat(self, grd_f, grd_c, R_FL, T_FL, shift_u, shift_v, theta, level, camera):
        '''
        Args:
            grd_f: [B, C, H, W]
            grd_c: [B, 1, H, W]
            R_FL: [B, 3, 3]
            T_FL: [B, 3]
            shift_u: [B, 1]
            shift_v: [B, 1]
            theta: [B, 1]
            level: scalar, feature level
        '''

        uv, mask = self.World2GrdImgPixCoordinates(R_FL, T_FL, shift_u, shift_v, theta, level, camera)

        grd_f_trans, _ = grid_sample(grd_f, uv, jac=None)
        # [B, C, H, W], [3, B, C, H, W]

        grd_f_trans = grd_f_trans * mask[:, None, :, :]

        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)  # [B, 1, H, W]
            grd_c_trans = grd_c_trans * mask[:, None, :, :]
        else:
            grd_c_trans = None

        return grd_f_trans, grd_c_trans, uv * mask[..., None], mask


    def project_grds_to_sat(self, grd_fs, grd_c, Rs, Ts, shift_u, shift_v, theta, level, cameras):
        grd_feat_cameras_proj = []
        grd_c_cameras = []
        grd_uvs_cameras = []
        masks_cameras = []
        for idx in range(len(cameras)):
            grd_feat_camera = grd_fs[idx]
            grd_feat_camera_proj, grd_c_camera, grd_uvs_camera, masks_camera = self.project_grd_to_sat(grd_feat_camera, grd_c, Rs, Ts, shift_u, shift_v, theta, level, cameras[idx])
            grd_feat_cameras_proj.append(grd_feat_camera_proj)
            grd_c_cameras.append(grd_c_camera)
            grd_uvs_cameras.append(grd_uvs_camera)
            masks_cameras.append(masks_camera)
        return grd_feat_cameras_proj, grd_c_cameras, grd_uvs_cameras, masks_cameras


    def Trans_update(self, shift_u, shift_v, theta, grd_feat_proj, sat_feat):
        B = shift_u.shape[0]
        grd_feat_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, 1e-6 * torch.ones_like(grd_feat_norm))
        grd_feat_proj = grd_feat_proj / grd_feat_norm[:, None, None, None]

        delta = self.TransRefine(grd_feat_proj, sat_feat)  # [B, 3]

        shift_u_new = shift_u + delta[:, 0:1]
        shift_v_new = shift_v + delta[:, 1:2]
        heading_new = theta + delta[:, 2:3]

        B = shift_u.shape[0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)
        # shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        # shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)

        return shift_u_new, shift_v_new, heading_new

    def corr(self, grd_imgs, sat_uncer_list, sat_feat_list, grd_feat_list, gt_shift_u=None, gt_shift_v=None, gt_heading=None, mode='train'):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:
        '''

        B, _, ori_grdH, ori_grdW = grd_imgs[0].shape

        corr_maps = []
        # print("agrs: ",args)
        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]

            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            A = sat_feat.shape[-1]

            crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(grd_feat, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]

            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            if self.args.use_uncertainty:
                denominator = torch.sum(denominator, dim=1) * TF.center_crop(sat_uncer_list[level], [corr.shape[1], corr.shape[2]])[:, 0]
            else:
                denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            # denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)
            # print("meter per pixel: ",self.meters_per_pixel[-1])

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = (max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading * self.args.rotation_range / 180 * np.pi)
            sin = torch.sin(gt_heading * self.args.rotation_range / 180 * np.pi)

            pred_u1 = - pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin - pred_v * cos


        self.corr_maps = corr_maps
        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u1, pred_v1  # [B], [B]
    

    def CrossAttn_rot_corr(self, sat_map, grd_imgs, gt_shift_u=None, gt_shift_v=None, gt_theta=None, mode='train', epoch=None, grd_name=None):

        B, _, ori_grdH, ori_grdW = grd_imgs[0].shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)
        if self.args.use_uncertainty:
            sat_uncer_list = self.uncertain_net(sat_feat_list)

        #concat along batch dim 
        grd_imgs_cat = torch.cat(grd_imgs, dim=0)
        grd8, grd4, grd2 = self.GrdEnc(grd_imgs_cat)
        grd_feat_cameras_list = self.GrdDec(grd8, grd4, grd2)
        #unconcat along batch dim
        #[feature_level, camera, [B, C, W, H]]
        grd_feat_cameras_list = [torch.chunk(grd_feat, len(self.args.cameras), dim=0) for grd_feat in grd_feat_cameras_list]

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        theta = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        shift_us_all = []
        shift_vs_all = []
        thetas_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            thetas = []
            for level in range(len(sat_feat_list)):
                sat_feat = sat_feat_list[level]
                grd_feat_cameras = grd_feat_cameras_list[level]

                grd_feat_cameras_proj, grd_conf_cameras_proj, grd_uv_cameras, mask_cameras = self.project_grds_to_sat(grd_feat_cameras, None, self.args.Rs, self.args.Ts, shift_u, shift_v, theta, level, self.args.cameras)
                grd_feat_proj = sum(grd_feat_cameras_proj)
                if self.args.lifting != 'homography':
                    grd_feat_proj = self.decoder_3dto2d_nets[level](grd_feat_proj)
                shift_u_new, shift_v_new, theta_new = self.Trans_update(shift_u, shift_v, theta, grd_feat_proj, sat_feat)

                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                thetas.append(theta_new[:, 0])  # [B]

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                theta = theta_new.clone()

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            thetas_all.append(torch.stack(thetas, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(thetas_all, dim=1)  # [B, N_iters, Level]

        # def cor(): used to be here 

        if mode == 'train':

            loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
            shift_lat_last, shift_lon_last, theta_last, \
                = loss_func(shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                            torch.exp(-self.coe_R), torch.exp(-self.coe_R), torch.exp(-self.coe_R))

            shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
            shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

            grd2sat8s, _, grd_uvs, masks = self.project_grds_to_sat(grd_feat_cameras_list[0], None, self.args.Rs, self.args.Ts, shift_u, shift_v, thetas[:, -1, -1:], level=0, cameras=self.args.cameras)
            grd2sat4s, _, _, _ = self.project_grds_to_sat(grd_feat_cameras_list[1], None, self.args.Rs, self.args.Ts, shift_u, shift_v, thetas[:, -1, -1:], level=1, cameras=self.args.cameras)
            grd2sat2s, _, _, _ = self.project_grds_to_sat(grd_feat_cameras_list[2], None, self.args.Rs, self.args.Ts, shift_u, shift_v, thetas[:, -1, -1:], level=2, cameras=self.args.cameras)

            #unconcat along batch dim
            if self.args.lifting == 'homography':
                grd8s = torch.chunk(grd8, len(self.args.cameras), dim=0) 
                grd2sat8_attn_list = [self.CVattn(grd2sat8s[idx], grd8s[idx], grd_uvs[idx][:, :, :, 0], masks[idx][..., None]) for idx, _ in enumerate(self.args.cameras)]
                grd2sat8_attn = sum(grd2sat8_attn_list)

                grd2sat4 = sum(grd2sat4s)
                grd2sat4_attn = grd2sat4 + self.Dec4(grd2sat8_attn, grd2sat4)
                grd2sat2 = sum(grd2sat2s)
                grd2sat2_attn = grd2sat2 + self.Dec2(grd2sat4_attn, grd2sat2)

                grd_feat_proj_list = [grd2sat8_attn, grd2sat4_attn, grd2sat2_attn]
            else:
                grd2sat8s = self.decoder_3dto2d_nets[0](sum(grd2sat8s))
                grd2sat4s = self.decoder_3dto2d_nets[1](sum(grd2sat4s))
                grd2sat2s = self.decoder_3dto2d_nets[2](sum(grd2sat2s))
                grd_feat_proj_list = [grd2sat8s, grd2sat4s, grd2sat2s]

            trans_loss = self.corr(grd_imgs, sat_uncer_list, sat_feat_list, grd_feat_proj_list, gt_shift_u, gt_shift_v, gt_theta, mode)

            return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                   shift_lat_last, shift_lon_last, theta_last, \
                   trans_loss
        else:

            shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
            shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

            grd2sat8s, _, grd_uvs, masks = self.project_grds_to_sat(grd_feat_cameras_list[0], None, self.args.Rs, self.args.Ts, shift_u, shift_v, thetas[:, -1, -1:], level=0, cameras=self.args.cameras)
            grd2sat4s, _, _, _ = self.project_grds_to_sat(grd_feat_cameras_list[1], None, self.args.Rs, self.args.Ts, shift_u, shift_v, thetas[:, -1, -1:], level=1, cameras=self.args.cameras)
            grd2sat2s, _, _, _ = self.project_grds_to_sat(grd_feat_cameras_list[2], None, self.args.Rs, self.args.Ts, shift_u, shift_v, thetas[:, -1, -1:], level=2, cameras=self.args.cameras)

            if self.args.lifting == 'homography':
                #unconcat along batch dim
                grd8s = torch.chunk(grd8, len(self.args.cameras), dim=0) 
                grd2sat8_attn_list = [self.CVattn(grd2sat8s[idx], grd8s[idx], grd_uvs[idx][:, :, :, 0], masks[idx][..., None]) for idx, _ in enumerate(self.args.cameras)]
                grd2sat8_attn = sum(grd2sat8_attn_list)

                grd2sat4 = sum(grd2sat4s)
                grd2sat4_attn = grd2sat4 + self.Dec4(grd2sat8_attn, grd2sat4)
                grd2sat2 = sum(grd2sat2s)
                grd2sat2_attn = grd2sat2 + self.Dec2(grd2sat4_attn, grd2sat2)

                grd_feat_proj_list = [grd2sat8_attn, grd2sat4_attn, grd2sat2_attn] 
            else:
                grd2sat8s = self.decoder_3dto2d_nets[0](sum(grd2sat8s))
                grd2sat4s = self.decoder_3dto2d_nets[1](sum(grd2sat4s))
                grd2sat2s = self.decoder_3dto2d_nets[2](sum(grd2sat2s))
                grd_feat_proj_list = [grd2sat8s, grd2sat4s, grd2sat2s]

            pred_u, pred_v = self.corr(grd_imgs, sat_uncer_list, sat_feat_list, grd_feat_proj_list, gt_shift_u, gt_shift_v, thetas[:, -1, -1:], mode) #use thetas[:, -1, -1:] instead of gt_theta in test mode 

            pred_orien = thetas[:, -1, -1]

            ############################### only used for presentation ##################################
            theta_zero = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
            grd2sat2s, _, _, _ = self.project_grds_to_sat(grd_feat_cameras_list[2], None, self.args.Rs, self.args.Ts, shift_u, shift_v, theta_zero, level=2, cameras=self.args.cameras)
            grd2sat2s = self.decoder_3dto2d_nets[2](sum(grd2sat2s))
            self.return_for_presentation = [sat_uncer_list[-1], grd2sat2s]

            return pred_u, pred_v, pred_orien#, grd_feat_proj_list, grd2sat8_attn_list
        

    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v, gt_heading):
        cos = torch.cos(gt_heading * self.args.rotation_range / 180 * np.pi)
        sin = torch.sin(gt_heading * self.args.rotation_range / 180 * np.pi)

        gt_delta_x = gt_shift_u * self.args.shift_range_lon
        gt_delta_y = gt_shift_v * self.args.shift_range_lat

        gt_delta_x_rot = - gt_delta_x * cos - gt_delta_y * sin
        gt_delta_y_rot = gt_delta_x * sin - gt_delta_y * cos

        losses = []
        for level in range(len(corr_maps)):
            meter_per_pixel = self.meters_per_pixel[level]

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
            h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            # import pdb; pdb.set_trace()
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))



def loss_func(shift_lats, shift_lons, thetas,
              gt_shift_lat, gt_shift_lon, gt_theta,
              coe_shift_lat=100, coe_shift_lon=100, coe_theta=100):
    '''
    Args:
        loss_method:
        ref_feat_list:
        pred_feat_dict:
        gt_feat_dict:
        shift_lats: [B, N_iters, Level]
        shift_lons: [B, N_iters, Level]
        thetas: [B, N_iters, Level]
        gt_shift_lat: [B]
        gt_shift_lon: [B]
        gt_theta: [B]
        pred_uv_dict:
        gt_uv_dict:
        coe_shift_lat:
        coe_shift_lon:
        coe_theta:
        coe_L1:
        coe_L2:
        coe_L3:
        coe_L4:

    Returns:

    '''
    B = gt_shift_lat.shape[0]
    # shift_lats = torch.stack(shift_lats_all, dim=1)  # [B, N_iters, Level]
    # shift_lons = torch.stack(shift_lons_all, dim=1)  # [B, N_iters, Level]
    # thetas = torch.stack(thetas_all, dim=1)  # [B, N_iters, Level]

    shift_lat_delta0 = torch.abs(shift_lats - gt_shift_lat[:, None, None])  # [B, N_iters, Level]
    shift_lon_delta0 = torch.abs(shift_lons - gt_shift_lon[:, None, None])  # [B, N_iters, Level]
    thetas_delta0 = torch.abs(thetas - gt_theta[:, None, None])  # [B, N_iters, level]

    shift_lat_delta = torch.mean(shift_lat_delta0, dim=0)  # [N_iters, Level]
    shift_lon_delta = torch.mean(shift_lon_delta0, dim=0)  # [N_iters, Level]
    thetas_delta = torch.mean(thetas_delta0, dim=0)  # [N_iters, level]

    shift_lat_decrease = shift_lat_delta[0] - shift_lat_delta[-1]  # [level]
    shift_lon_decrease = shift_lon_delta[0] - shift_lon_delta[-1]  # [level]
    thetas_decrease = thetas_delta[0] - thetas_delta[-1]  # [level]

    losses = coe_shift_lat * shift_lat_delta + coe_shift_lon * shift_lon_delta + coe_theta * thetas_delta  # [N_iters, level]
    loss_decrease = losses[0] - losses[-1]  # [level]
    loss = torch.mean(losses)  # mean or sum
    loss_last = losses[-1]

    return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
            shift_lat_delta[-1], shift_lon_delta[-1], thetas_delta[-1]
        


# def normalize_feature(x):
#     C, H, W = x.shape[-3:]
#     norm = torch.norm(x.flatten(start_dim=-3), dim=-1)
#     return x / norm[..., None, None, None]





    # def rot_corr(self, sat_map, grd_img_left, R_FL, T_FL, gt_shift_u=None, gt_shift_v=None, gt_theta=None, mode='train', epoch=None):
    #     '''
    #     :param sat_map: [B, C, A, A] A--> sidelength
    #     :param grd_img_left: [B, C, H, W]
    #     :return:
    #     '''

    #     B, _, ori_grdH, ori_grdW = grd_img_left.shape

    #     sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)
    #     if self.args.use_uncertainty:
    #         sat_uncer_list = self.uncertain_net(sat_feat_list)

    #     grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

    #     shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
    #     shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
    #     theta = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

    #     shift_us_all = []
    #     shift_vs_all = []
    #     thetas_all = []
    #     for iter in range(self.N_iters):
    #         shift_us = []
    #         shift_vs = []
    #         thetas = []
    #         for level in range(len(sat_feat_list)):
    #             sat_feat = sat_feat_list[level]
    #             grd_feat = grd_feat_list[level]

    #             grd_feat_proj, _, grd_uv, mask = self.project_grd_to_sat(
    #                 grd_feat, None, R_FL, T_FL, shift_u, shift_v, theta, level)
    #             # [B, C, H, W], [B, 1, H, W], [B, H, W, 2]

    #             shift_u_new, shift_v_new, theta_new = self.Trans_update(shift_u, shift_v, theta,
    #                                                                  grd_feat_proj,
    #                                                                  sat_feat)

    #             shift_us.append(shift_u_new[:, 0])  # [B]
    #             shift_vs.append(shift_v_new[:, 0])  # [B]
    #             thetas.append(theta_new[:, 0])  # [B]

    #             shift_u = shift_u_new.clone()
    #             shift_v = shift_v_new.clone()
    #             theta = theta_new.clone()

    #         shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
    #         shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
    #         thetas_all.append(torch.stack(thetas, dim=1))  # [B, Level]

    #     shift_lats = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
    #     shift_lons = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
    #     thetas = torch.stack(thetas_all, dim=1)  # [B, N_iters, Level]

    #     def corr(sat_feat_list, grd_feat_list, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
    #              pred_heading=None, mode='train'):

    #         B, _, ori_grdH, ori_grdW = grd_img_left.shape

    #         shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
    #         shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

    #         corr_maps = []

    #         for level in range(len(sat_feat_list)):
    #             meter_per_pixel = self.meters_per_pixel[level]

    #             sat_feat = sat_feat_list[level]
    #             grd_feat = grd_feat_list[level]

    #             A = sat_feat.shape[-1]
    #             if mode == 'train':
    #                 theta = gt_heading[:, None] #+ np.random.uniform(-0.1, 0.1)
    #             else:
    #                 theta = pred_heading

    #             grd_feat_proj, _, grd_uv, mask = self.project_grd_to_sat(
    #                 grd_feat, None, R_FL, T_FL, shift_u, shift_v, theta, level)
    #             # [B, C, H, W], [B, 1, H, W], [3, B, C, H, W], [B, H, W, 2]

    #             crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
    #             crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
    #             g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
    #             g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

    #             s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]

    #             corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

    #             denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
    #             if self.args.use_uncertainty:
    #                 denominator = torch.sum(denominator, dim=1) * TF.center_crop(sat_uncer_list[level], [corr.shape[1], corr.shape[2]])[:, 0]
    #             else:
    #                 denominator = torch.sum(denominator, dim=1)  # [B, H, W]

    #             denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
    #             corr = 2 - 2 * corr / denominator

    #             B, corr_H, corr_W = corr.shape

    #             corr_maps.append(corr)

    #             max_index = torch.argmin(corr.reshape(B, -1), dim=1)
    #             pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
    #             pred_v = (max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

    #             cos = torch.cos(gt_heading * self.args.rotation_range / 180 * np.pi)
    #             sin = torch.sin(gt_heading * self.args.rotation_range / 180 * np.pi)

    #             pred_u1 = - pred_u * cos + pred_v * sin
    #             pred_v1 = - pred_u * sin - pred_v * cos

    #         if mode == 'train':
    #             return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
    #         else:
    #             return pred_u1, pred_v1  # [B], [B]

    #     if mode == 'train':

    #         if self.args.rotation_range == 0:
    #             coe_heading = 0
    #         else:
    #             coe_heading = self.args.coe_heading

    #         loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
    #         shift_lat_last, shift_lon_last, theta_last, \
    #             = loss_func(shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
    #                         self.args.coe_shift_lat, self.args.coe_shift_lon, coe_heading)

    #         trans_loss = corr(sat_feat_list, grd_feat_list, gt_shift_u, gt_shift_v, gt_theta,
    #                           thetas[:, -1, -1:], mode)

    #         return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
    #                shift_lat_last, shift_lon_last, theta_last, \
    #                grd_conf_list, trans_loss
    #     else:
    #         pred_u, pred_v = corr(sat_feat_list, grd_feat_list, gt_shift_u, gt_shift_v, gt_theta,
    #                               thetas[:, -1, -1:], mode)
    #         pred_orien = thetas[:, -1, -1]

    #         return pred_u, pred_v, pred_orien

