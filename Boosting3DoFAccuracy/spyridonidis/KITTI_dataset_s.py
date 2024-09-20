import os
os.chdir('../')
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.optim as optim
from dataLoader.KITTI_dataset import load_train_data, load_test1_data, load_test2_data
# from models_kitti import Model
import scipy.io as scio
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

import numpy as np
import os
import argparse
import time


from PIL import Image
from torch.utils.data import Dataset

import torch
import pandas as pd
import utils
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms



root_dir = '../../../datasets/KITTI'
test_csv_file_name = 'test.csv'
ignore_csv_file_name = 'ignore.csv'
satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'  # 'image_02\\data' #
# right_color_camera_dir = 'image_03/data'  # 'image_03\\data' #
oxts_dir = 'oxts/data'  # 'oxts\\data' #
# depth_dir = 'depth/data_depth_annotated/train/'

GrdImg_H = 256  # 256 # original: 375 #224, 256
GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 375
GrdOriImg_W = 1242
num_thread_workers = 2

test1_file = './dataLoader/test1_files.txt'

SatMap_process_sidelength = utils.get_process_satmap_sidelength()

satmap_transform = transforms.Compose([
    transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
    transforms.ToTensor(),
])

Grd_h = GrdImg_H
Grd_w = GrdImg_W

grdimage_transform = transforms.Compose([
    transforms.Resize(size=[Grd_h, Grd_w]),
    transforms.ToTensor(),
])

def getSavePath(args):
    save_path = './ModelsKitti/3DoF/'\
                + 'lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range) \
                + '_Nit' + str(args.N_iters) + '_' + str(args.Optimizer) + '_' + str(args.proj)

    if args.use_uncertainty:
        save_path = save_path + '_Uncertainty'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('save_path:', save_path)

    return save_path


def parse_args(args_input = ['--batch_size', '3']):
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 1e-2

    parser.add_argument('--rotation_range', type=float, default=10., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--batch_size', type=int, default=3, help='batch size')

    parser.add_argument('--level', type=int, default=3, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--N_iters', type=int, default=2, help='any integer')

    parser.add_argument('--Optimizer', type=str, default='TransV1G2SP', help='')

    parser.add_argument('--proj', type=str, default='CrossAttn', help='geo, CrossAttn')

    parser.add_argument('--use_uncertainty', type=int, default=1, help='0 or 1')

    args = parser.parse_args(args_input)# must pass at least one argument when using jupiter notebook 
    return args



class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        # self.shift_range_meters = shift_range  # in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        # np.random.seed(2022)
        # num = len(file_name)//3
        # random.shuffle(file_name)
        # self.file_name = [file[:-1] for file in file_name[:num]]
        self.file_name = [file[:-1] for file in file_name]
        # self.file_name = []
        # count = 0
        # for line in file_name:
        #     file = line.split(' ')[0]
        #     left_depth_name = os.path.join(self.root, depth_dir, file.split('/')[1],
        #                                    'proj_depth/groundtruth/image_02', os.path.basename(file.strip()))
        #     if os.path.exists(left_depth_name):
        #         self.file_name.append(line.strip())
        #     else:
        #         count += 1
        #
        # print('number of files whose depth unavailable: ', count)


    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        
        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        grd_left_depths = torch.tensor([])
        # image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            heading = float(content[5])
            heading = torch.from_numpy(np.asarray(heading))

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            with Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            # left_depth_name = os.path.join(self.root, depth_dir, file_name.split('/')[1],
            #                                'proj_depth/groundtruth/image_02', image_no)

            # left_depth = torch.tensor(depth_read(left_depth_name), dtype=torch.float32)
            # left_depth = F.interpolate(left_depth[None, None, :, :], (GrdImg_H, GrdImg_W))
            # left_depth = left_depth[0, 0]

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
            # grd_left_depths = torch.cat([grd_left_depths, left_depth.unsqueeze(0)], dim=0)

        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel ,
                                           0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel ),
                                           resample=Image.BILINEAR)
        
        test_data = TF.center_crop(sat_align_cam, utils.SatMap_process_sidelength)
        # the homography is defined on: from target pixel to source pixel
        # now east direction is the real vehicle heading direction

        # randomly generate shift
        # gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        # gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction or v cordinate 
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction or -u cordinate 

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon ,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)



        # randomly generate roation
        # theta = np.random.uniform(-1, 1)
        theta = float(theta)
        sat_rand_shift_rand_rot = \
            sat_rand_shift.rotate(theta * self.rotation_range)#rotate takes input degrees theta * self.rotation_range
        
        
        sat_map = TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)#crops from 1280 to 512 pixels that 
        # sat_map = np.array(sat_map, dtype=np.float32)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # gt_corr_x, gt_corr_y = self.generate_correlation_GTXY(gt_shift_x, gt_shift_y, theta)

        return sat_map, left_camera_k, grd_left_imgs[0], \
               torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1), \
               torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1), \
               torch.tensor(theta, dtype=torch.float32).reshape(1), \
               file_name, \
               TF.to_tensor(test_data)
    
