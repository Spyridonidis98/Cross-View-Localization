import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
import pandas as pd
import utils
import torchvision.transforms.functional as TF
from torchvision import transforms
from cfgnode import CfgNode
import yaml

# 
import argparse


# Ford_root = '/media/yujiao/6TB/dataset/Ford/'
Ford_root = '../../../datasets/Ford/'
satmap_dir = 'SatelliteMaps_18'
data_file = 'grd_sat_quaternion_latlon.txt'
data_file_test = 'grd_sat_quaternion_latlon_test.txt'
pose_file_dir = 'Calibration-V2/V2/'
cameras_ex = {'FL':'cameraFrontLeft_body.yaml', 'SL': 'cameraSideLeft_body.yaml', 'SR': 'cameraSideRight_body.yaml', 'RL':'cameraRearLeft_body.yaml'}
cameras_in = {'FL':'cameraFrontLeftIntrinsics.yaml', 'SL': 'cameraSideLeftIntrinsics.yaml', 'SR':'cameraSideRightIntrinsics.yaml', 'RL':'cameraRearLeftIntrinsics.yaml'} 

train_logs = [
              '2017-10-26/V2/Log1',
              '2017-10-26/V2/Log2',
              '2017-08-04/V2/Log3',
              '2017-10-26/V2/Log4',
              '2017-08-04/V2/Log5',
              '2017-08-04/V2/Log6',
              ]

train_logs_img_inds = [
    list(range(4500, 8500)),
    list(range(3150)) + list(range(6000, 9200)) + list(range(11000, 15000)),
    list(range(1500)),
    list(range(7466)),
    list(range(3200)) + list(range(5300, 9900)) + list(range(10500, 11130)),
    list(range(1000, 3500)) + list(range(4500, 5000)) + list(range(7000, 7857)),
                       ]

test_logs = [
             '2017-08-04/V2/Log1',
             '2017-08-04/V2/Log2',
             '2017-08-04/V2/Log3',
             '2017-08-04/V2/Log4',
             '2017-10-26/V2/Log5',
             '2017-10-26/V2/Log6',
]
test_logs_img_inds = [
    list(range(100, 200)) + list(range(5000, 5500)) + list(range(7000, 8500)),
    list(range(2500, 3000)) + list(range(8500, 10500)) + list(range(12500, 13727)),
    list(range(3500, 5000)),
    list(range(1500, 2500)) + list(range(4000, 4500)) + list(range(7000, 9011)),
    list(range(3500)),
    list(range(2000, 2500)) + list(range(3500, 4000)),
]

# For the Ford dataset coordinates:
# x--> North, y --> east, z --> down
# North direction as 0-degree, clockwise as positive.
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 1e-2

    parser.add_argument('--rotation_range', type=float, default=10., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    parser.add_argument('--level', type=int, default=3, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--N_iters', type=int, default=2, help='any integer')

    parser.add_argument('--train_log_start', type=int, default=0, help='')
    parser.add_argument('--train_log_end', type=int, default=1, help='')
    parser.add_argument('--test_log_ind', type=int, default=0, help='')

    parser.add_argument('--proj', type=str, default='CrossAttn', help='geo, polar, nn, CrossAttn')

    parser.add_argument('--train_whole', type=int, default=0, help='0 or 1')
    parser.add_argument('--test_whole', type=int, default=0, help='0 or 1')

    parser.add_argument('--use_uncertainty', type=int, default=1, help='0 or 1')

    args = parser.parse_args(args)

    return args

def getSavePath(args):
    save_path = './ModelsFord/3DoF/' \
                + 'Log_' + str(args.train_log_start+1) + 'lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range) \
                + '_Nit' + str(args.N_iters) + '_' + str(args.proj)

    if args.use_uncertainty:
        save_path = save_path + '_Uncertainty'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('save_path:', save_path)

    return save_path


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def qvec2angle(q0, q1, q2, q3):
    roll  = np.arctan2(2.0 * (q3 * q2 + q0 * q1) , 1.0 - 2.0 * (q1 * q1 + q2 * q2)) / np.pi * 180
    pitch = np.arcsin(2.0 * (q2 * q0 - q3 * q1)) / np.pi * 180
    yaw   = np.arctan2(2.0 * (q3 * q0 + q1 * q2) , - 1.0 + 2.0 * (q0 * q0 + q1 * q1)) / np.pi * 180
    return roll, pitch, yaw


class SatGrdDatasetFord(Dataset):
    def __init__(self, root=Ford_root, logs=train_logs, logs_img_inds=train_logs_img_inds,
                 shift_range_lat=20, shift_range_lon=20, rotation_range=20, whole=False, H = 448, W = 896, cameras = ['FL'], mode = 'train'):
        self.root = root
        self.cameras = cameras
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.meters_per_pixel = 0.22
        self.shift_range_pixels_lat = shift_range_lat / self.meters_per_pixel  # in terms of pixels
        self.shift_range_pixels_lon = shift_range_lon / self.meters_per_pixel  # in terms of pixels
        self.mode = mode

        self.rotation_range = rotation_range # in terms of degree

        self.satmap_dir = satmap_dir
        self.lat0 = 42.29424422604817  # 08-04-Log0-img0

        self.H_ori = 860 # original image dimenstions 
        self.W_ori = 1656
        self.H = H # used by authors of bosting3dof # self.H = 256  # self.W = 1024
        self.W = W

        self.file_names = {}
        self.Ks = {}
        self.Rs = {}
        self.Ts = {}
        df = data_file if mode == 'train' else data_file_test
        for camera in cameras:
            file_name = []
            for idx in range(len(logs)):
                log = logs[idx]
                img_inds = logs_img_inds[idx]
                FL_dir = os.path.join(root, log, log.replace('/', '-') + '-' + camera)

                with open(os.path.join(root, log, df), 'r') as f:
                    lines = f.readlines()
                    if mode == 'train':
                        if whole == 0:
                            lines = [lines[ind] for ind in img_inds]
                    # lines = f.readlines()[img_inds]
                    for line in lines:
                        if mode == 'train':
                            grd_name, q0, q1, q2, q3, g_lat, g_lon, s_lat, s_lon = line.strip().split(' ')
                        else:
                            grd_name, q0, q1, q2, q3, g_lat, g_lon, s_lat, s_lon, gt_shift_u, gt_shift_v, theta = line.strip().split(' ')
                        grd_file_FL = os.path.join(FL_dir, grd_name.replace('.txt', '.png'))
                        sat_file = os.path.join(root, log, satmap_dir, s_lat + '_' + s_lon + '.png')
                        if mode == 'train':
                            file_name.append([grd_file_FL, float(q0), float(q1), float(q2), float(q3), float(g_lat), float(g_lon),
                                        float(s_lat), float(s_lon), sat_file])
                        else:
                            file_name.append([grd_file_FL, float(q0), float(q1), float(q2), float(q3), float(g_lat), float(g_lon),
                                        float(s_lat), float(s_lon), sat_file, 
                                        float(gt_shift_u), float(gt_shift_v),float(theta)])
                            
            self.file_names[camera] = file_name

            with open(os.path.join(root, pose_file_dir, cameras_ex[camera]), "r") as f:
                cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
                cfg_ex = CfgNode(cfg_dict)

            qx = cfg_ex.transform.rotation.x
            qy = cfg_ex.transform.rotation.y
            qz = cfg_ex.transform.rotation.z
            qw = cfg_ex.transform.rotation.w

            FLx, FLy, FLz = cfg_ex.transform.translation.x, cfg_ex.transform.translation.y, cfg_ex.transform.translation.z
            self.Ts[camera] = np.array([FLx, FLy, FLz]).reshape(3).astype(np.float32)
            self.Rs[camera] = qvec2rotmat([qw, qx, qy, qz]).astype(np.float32)
            # from camera coordinates to body coordinates
            # Xb = Rs[camera] @ Xc + Ts[camera]

            with open(os.path.join(root, pose_file_dir, cameras_in[camera]), "r") as f:
                cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
                cfg_FL_in = CfgNode(cfg_dict)
            self.Ks[camera] = np.array(cfg_FL_in.K, dtype=np.float32).reshape([3, 3])
        
            self.Ks[camera][0] = self.Ks[camera][0] / self.W_ori * self.W
            self.Ks[camera][1] = self.Ks[camera][1] / self.H_ori * self.H

        self.sidelength = 512
        self.satmap_sidelength_meters = self.sidelength * self.meters_per_pixel
        self.satmap_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.grdimage_transform = transforms.Compose([
            transforms.Resize(size=[self.H, self.W]),
            transforms.ToTensor(),
        ])

        self.Rs = {key:torch.tensor(item, dtype=torch.float32) for key, item in self.Rs.items()}
        self.Ts = {key:torch.tensor(item, dtype=torch.float32) for key, item in self.Ts.items()}
        self.Ks = {key:torch.tensor(item, dtype=torch.float32) for key, item in self.Ks.items()}

    def __len__(self):
        return len(list(self.file_names.values())[0])

    def get_file_list(self):
        return self.file_names

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name
        grd_imgs = []; grd_names = []
        first_cam = tuple(self.file_names.keys())[0]

        if self.mode == 'train':
            _, q0, q1, q2, q3, g_lat, g_lon, s_lat, s_lon, sat_name = self.file_names[first_cam][idx]
        else:
            _, q0, q1, q2, q3, g_lat, g_lon, s_lat, s_lon, sat_name, gt_shift_u, gt_shift_v, theta = self.file_names[first_cam][idx]

        # Xc = np.array([0, 0, 0]).reshape(3)
        # Rw = qvec2rotmat([float(q0), float(q1), float(q2), float(q3)])
        # # body frame to world frame: Xw = Rw @ Xb + Tw  (Tw are all zeros)
        # Xw = Rw @ (self.R_FL @ Xc + self.T_FL)  # North (up) --> X, East (right) --> Y
        # # camera location represented in world coordinates,
        # # world coordinates is centered at the body coordinates, but with X pointing north, Y pointing east, Z pointing down

        g_x, g_y = utils.gps2utm(float(g_lat), float(g_lon), float(s_lat))
        s_x, s_y = utils.gps2utm(float(s_lat), float(s_lon), float(s_lat))
        # x, y here are the x, y under gps/utm coordinates, x pointing right and y pointing up

        b_delta_u = (g_x - s_x) / self.meters_per_pixel # relative u shift of body frame with respect to satellite image center
        b_delta_v = - (g_y - s_y) / self.meters_per_pixel # relative v shift of body frame with respect to satellite image center

        sat_map = Image.open(sat_name).convert('RGB')
        sat_align_body_loc = sat_map.transform(sat_map.size, Image.AFFINE,
                                        (1, 0, b_delta_u,
                                        0, 1, b_delta_v),
                                        resample=Image.BILINEAR)
        # Homography is defined on from target pixel to source pixel
        roll, pitch, yaw = qvec2angle(q0, q1, q2, q3)  # in terms of degree
        sat_align_body_loc_orien = sat_align_body_loc.rotate(yaw)

        # random shift
        if self.mode == 'train':
            gt_shift_u = np.random.uniform(-1, 1)  # --> right (east) as positive, vertical to the heading, lateral
            gt_shift_v = np.random.uniform(-1, 1)  # --> down (south) as positive, parallel to the heading, longitudinal

        sat_rand_shift = \
            sat_align_body_loc_orien.transform(
                sat_align_body_loc_orien.size, Image.AFFINE,
                (1, 0, gt_shift_u * self.shift_range_pixels_lat,
                0, 1, gt_shift_v * self.shift_range_pixels_lon),
                resample=Image.BILINEAR)

        if self.mode == 'train':
            theta = np.random.uniform(-1, 1)
        sat_rand_shift_rot = sat_rand_shift.rotate(theta * self.rotation_range)

        sat_img = TF.center_crop(sat_rand_shift_rot, self.sidelength)
        sat_img = self.satmap_transform(sat_img)

        for camera in self.cameras:
            if self.mode == 'train':
                grd_name, _, _, _, _, _, _, _, _, _ = self.file_names[camera][idx]
            else:
                grd_name, _, _, _, _, _, _, _, _, _, _, _, _ = self.file_names[camera][idx]
                
            grd_img = Image.open(grd_name).convert('RGB')
            grd_img = self.grdimage_transform(grd_img)

            grd_imgs.append(grd_img)
            grd_names.append(grd_name)

        sat_img_norot_notrans = TF.center_crop(sat_align_body_loc_orien, self.sidelength)
        sat_img_norot_notrans = self.satmap_transform(sat_img_norot_notrans)


        gt_shift_u = torch.tensor(gt_shift_u, dtype=torch.float32)
        gt_shift_v = torch.tensor(gt_shift_v, dtype=torch.float32)
        theta = torch.tensor(theta, dtype=torch.float32)
        xy_dt_mask, xy_dt = satimgtrans2satimgorig(gt_shift_v, gt_shift_u, theta, self.shift_range_meters_lat, self.shift_range_meters_lon, self.rotation_range, self.Rs, self.Ts, self.Ks, self.H, self.W)

        return sat_img,\
               tuple(grd_imgs),\
               gt_shift_u,\
               gt_shift_v,\
               theta,\
               tuple(grd_names), sat_img_norot_notrans, xy_dt_mask, xy_dt
               #self.satmap_transform(sat_align_body_loc_orien), self.satmap_transform(sat_rand_shift), self.satmap_transform(sat_rand_shift_rot) 
               

def get_xy_map(sidelength):
    # shape: (2, sidelength, sidelength)
    meters_range = (torch.arange(sidelength) / (sidelength - 1))-0.5
    meters_y = meters_range.repeat(sidelength, 1)
    meters_x = meters_y.transpose(0,1)
    meters_xy = torch.stack((meters_x, meters_y), dim=0) # [-0.5, 0.5] representing [-56.3, 56.3] meters of the satelite map 

    return meters_xy

def get_xyz_map(sidelength):
    # x,y axis range [-56.3, 56.3] 
    # shape: (sidelength, sidelength, 1, 3)

    x = torch.arange(sidelength).flip(dims=[0]) / (sidelength-1) - 0.5; x=x*56.3*2
    x = x.repeat(sidelength, 1).T

    y = torch.arange(sidelength) / (sidelength-1) -0.5; y=y*56.3*2
    y = y.repeat(sidelength, 1)

    z = torch.zeros_like(x); z[:,:] = 1.4 #for z=+1.4 we are at the level of the ground

    xyz = torch.stack((x, y, z), dim=-1)
    
    return xyz[:,:,None,:]

def get_xyz_ids(sidelength):
    # shape: (sidelength, sidelength, 1, 3)
    x_id = torch.arange(sidelength) 
    x_id = x_id.repeat(sidelength, 1).T

    y_id = torch.arange(sidelength) 
    y_id = y_id.repeat(sidelength, 1)

    z_id = torch.zeros_like(x_id)

    xyz_ids = torch.stack((x_id, y_id, z_id), dim=-1)

    return xyz_ids[:,:,None,:]

def render_xyz(R, T, K, xyz, xyz_ids, H=448, W=896):
    '''
    scatter_ids 3,n
    uvs:2,n
    '''
    xyz_cam = R.T @ xyz.view(-1, 3).T - T.unsqueeze(1)
    uvs = K @ xyz_cam
    uvs[0] = uvs[0]/ uvs[2]
    uvs[1] = uvs[1]/ uvs[2]

    bf = (uvs[0, :] >= 0) & (uvs[1, :] >= 0) & (uvs[0, :] < W-1) & (uvs[1, :]< H-1) & (uvs[2, :] > 0.1)
    uvs = uvs[:, bf]
    scatter_ids = xyz_ids.view(-1, 3).T[:, bf]

    return uvs, scatter_ids

def lift_features(img, volume, uvs, scatter_ids):
    gather_ids = torch.zeros_like(uvs)
    gather_ids[0, :] = uvs[1, :]
    gather_ids[1, :] = uvs[0, :]

    gather_ids_dl = gather_ids.clone() #down left 
    gather_ids_dl[0, :] = gather_ids_dl[0, :].floor()
    gather_ids_dl[1, :] = gather_ids_dl[1, :].floor()

    gather_ids_ul = gather_ids.clone() #up left 
    gather_ids_ul[0, :] = gather_ids_ul[0, :].floor() + 1.0
    gather_ids_ul[1, :] = gather_ids_ul[1, :].floor()

    gather_ids_ur = gather_ids.clone() #up right
    gather_ids_ur[0, :] = gather_ids_ur[0, :].floor() + 1.0
    gather_ids_ur[1, :] = gather_ids_ur[1, :].floor() + 1.0

    gather_ids_dr = gather_ids.clone() #down right 
    gather_ids_dr[0, :] = gather_ids_dr[0, :].floor()
    gather_ids_dr[1, :] = gather_ids_dr[1, :].floor() + 1.0

    x_u = gather_ids_ul[0, :] #up 
    x_d = gather_ids_dl[0, :] #down
    y_r = gather_ids_dr[1, :] #right
    y_l = gather_ids_dl[1, :] #left 

    w_dl = (x_u - gather_ids[0]) * (y_r - gather_ids[1]) #/ (x_u - x_d) * (y_r - y_l) #down left weights, no need to divide since the volume is always 1  
    w_ul = (gather_ids[0] - x_d) * (y_r - gather_ids[1]) #/ (x_u - x_d) * (y_r - y_l) #up left weights 
    w_ur = (gather_ids[0] - x_d) * (gather_ids[1] - y_l) #/ (x_u - x_d) * (y_r - y_l) #up right weights 
    w_dr = (x_u - gather_ids[0]) * (gather_ids[1] - y_l) #/ (x_u - x_d) * (y_r - y_l) #down right weights 

    gather_ids_dl = gather_ids_dl.to(torch.int64)
    gather_ids_ul = gather_ids_ul.to(torch.int64)
    gather_ids_ur = gather_ids_ur.to(torch.int64)
    gather_ids_dr = gather_ids_dr.to(torch.int64)

    volume[:, scatter_ids[0], scatter_ids[1], scatter_ids[2]] =  w_dl * img[:, gather_ids_dl[0], gather_ids_dl[1]] \
                                                                + w_ul * img[:, gather_ids_ul[0], gather_ids_ul[1]] \
                                                                + w_ur * img[:, gather_ids_ur[0], gather_ids_ur[1]] \
                                                                + w_dr * img[:, gather_ids_dr[0], gather_ids_dr[1]] \
                                                                    
    return volume

def get_camera_mask(R, T, K, shift_x = 0.5, shift_y =0.5, theta = 45, H=448, W=896, sidelength = 64):
    # shift x,y in range [-0.5, 0.5]
    # theta in degrees
    xyz = get_xyz_map(sidelength)
    xyz[:,:,:,0] -= shift_x *56.3*2
    xyz[:,:,:,1] += shift_y *56.3*2
    
    theta_rad = theta * torch.pi/ 180 ;theta_rad = torch.tensor(theta_rad)
    rot_mat = torch.tensor([
            [torch.cos(theta_rad), -torch.sin(theta_rad),0],
            [torch.sin(theta_rad), torch.cos(theta_rad),0],
            [0, 0, 1]
        ])
    xyz_rot_trans = rot_mat @ xyz.view(-1, 3).T; xyz = xyz_rot_trans.T.view(*xyz.shape)
    
    xyz_ids = get_xyz_ids(sidelength)
    volume = torch.zeros(size=(1,*xyz.shape[:3]))
    img = torch.ones(size=(1, H, W))
    uvs, scatter_ids= render_xyz(R, T, K, xyz, xyz_ids, H, W)
    volume = lift_features(img, volume, uvs, scatter_ids); volume = volume[0,:,:,0].ceil().to(torch.bool)
    return volume


def satimgtrans2satimgorig(shift_x, shift_y, theta, range_lat = 20, range_lot = 20, rotation_range=20, Rs=None, Ts=None, Ks=None, H=448, W=896):
    """
    shift_x, shift_y, theta [-1,1] -> numpy.random
    """
    meters_per_pixel = 0.22 # for satelite image of sidelength 512
    sidelength_orig = 512
    sidelength = 64
    meters_xy = get_xy_map(sidelength)

    # create rotation and tranlation matrix
    shift_x_range = (shift_x * range_lat) / (meters_per_pixel * sidelength_orig) # [-0.5, 0.5] top left corner of the image is -0.5,-0.5
    shift_y_range = (shift_y * range_lot) / (meters_per_pixel * sidelength_orig) # [-0.5, 0.5] bottom left corner of the image is 0.5, -0.5
    theta_rad = theta * rotation_range * torch.pi/ 180 
    theta_rad = torch.tensor(theta_rad)

    rot_mat = torch.tensor([
            [torch.cos(theta_rad), torch.sin(theta_rad)],
            [-torch.sin(theta_rad), torch.cos(theta_rad)]
        ])
    
    trans_mat = torch.tensor([
        [shift_x_range],
        [shift_y_range]
    ])
    

    #rotate and translate points to the original potition
    meters_xy_orig = rot_mat @ meters_xy.view(2, -1) + trans_mat
    meters_xy_orig = meters_xy_orig.view(2, sidelength, sidelength)

    cameras_mask = None
    for camera in Rs.keys():
        R = Rs[camera]
        T = Ts[camera]
        K = Ks[camera]
        if cameras_mask is None:
            cameras_mask = get_camera_mask(R, T, K, shift_x_range, shift_y_range, theta * rotation_range, H, W, sidelength)
        else:
            cameras_mask = cameras_mask | get_camera_mask(R, T, K, shift_x_range, shift_y_range, theta * rotation_range, H, W, sidelength)

    meters_mask = (meters_xy_orig[0] <= 0.5) & (meters_xy_orig[0] >= -0.5) & (meters_xy_orig[1] <= 0.5) & (meters_xy_orig[1] >= -0.5) 
    meters_mask = meters_mask & cameras_mask
    meters_xy_dt = (meters_xy_orig - meters_xy) * meters_mask

    return meters_mask.long(), meters_xy_dt


