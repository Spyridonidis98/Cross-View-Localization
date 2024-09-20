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
cameras_ex = {'FL':'cameraFrontLeft_body.yaml', 'SL': 'cameraSideLeft_body.yaml'}
cameras_in = {'FL':'cameraFrontLeftIntrinsics.yaml', 'SL': 'cameraSideLeftIntrinsics.yaml'} 

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
                 shift_range_lat=20, shift_range_lon=20, rotation_range=10, whole=False, H = 448, W = 896, cameras = ['FL'], mode = 'train'):
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
        grd_imgs =[]; grd_names = []
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

        return sat_img, \
               tuple(grd_imgs), \
               torch.tensor(gt_shift_u, dtype=torch.float32), \
               torch.tensor(gt_shift_v, dtype=torch.float32), \
               torch.tensor(theta, dtype=torch.float32), \
               tuple(grd_names)


class SatGrdDatasetFordPresentation(Dataset):
    def __init__(self, root=Ford_root, logs=train_logs, logs_img_inds=train_logs_img_inds,
                 shift_range_lat=20, shift_range_lon=20, rotation_range=10, whole=False, H = 448, W = 896, cameras = ['FL'], mode = 'train'):
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
        rotation_circle = 100

        grd_imgs =[]; grd_names = []
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
            th = idx*2*np.pi/rotation_circle
            gt_shift_u = 0.6*np.cos(th)#np.random.uniform(-1, 1)  # --> right (east) as positive, vertical to the heading, lateral
            gt_shift_v = 0.3*np.sin(th) #np.random.uniform(-1, 1)  # --> down (south) as positive, parallel to the heading, longitudinal
            gt_shift_v += 0.7
            if gt_shift_u>0:
                gt_shift_u+=0.4
            else:
                gt_shift_u-=0.4

        sat_rand_shift = \
            sat_align_body_loc_orien.transform(
                sat_align_body_loc_orien.size, Image.AFFINE,
                (1, 0, gt_shift_u * self.shift_range_pixels_lat,
                0, 1, gt_shift_v * self.shift_range_pixels_lon),
                resample=Image.BILINEAR)

        if self.mode == 'train':
            theta = np.cos(th)
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


        return sat_img, \
               tuple(grd_imgs), \
               torch.tensor(gt_shift_u, dtype=torch.float32), \
               torch.tensor(gt_shift_v, dtype=torch.float32), \
               torch.tensor(theta, dtype=torch.float32), \
               tuple(grd_names), sat_img_norot_notrans, \
               torch.tensor(s_lat, dtype=torch.float64), torch.tensor(s_lon, dtype=torch.float64), torch.tensor(g_lat, dtype=torch.float64), torch.tensor(g_lon, dtype=torch.float64), \
               torch.tensor(yaw, dtype = torch.float32)
                 #self.satmap_transform(sat_align_body_loc_orien), self.satmap_transform(sat_rand_shift), self.satmap_transform(sat_rand_shift_rot) 
               