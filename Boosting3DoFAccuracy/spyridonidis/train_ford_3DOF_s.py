#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
print(os.getcwd())
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

import torch.optim as optim

from spyridonidis.Ford_dataset_s import SatGrdDatasetFord, test_logs, test_logs_img_inds, test_logs, test_logs_img_inds

import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

from spyridonidis.models_ford_s import ModelFord

import numpy as np
import os
import argparse

from torch.utils.data import DataLoader
import time

# for mixed presision training
from torch.cuda.amp import GradScaler
scaler = GradScaler()

torch.autograd.set_detect_anomaly(True)



def test1(net, args, save_path, testloader, epoch=0, device=torch.device("cuda:0")):

    net.eval()
    mini_batch = args.batch_size

    np.random.seed(2022)
    torch.manual_seed(2022)

    pred_shifts = []
    pred_headings = []
    gt_shifts = []
    gt_headings = []

    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            sat_img, grd_imgs, gt_shift_u, gt_shift_v, gt_heading, grd_names = [[camera.to(device) if type(camera) == torch.Tensor else camera for camera in item] if type(item)== list else item.to(device) if type(item) == torch.Tensor else item for item in data]
            if args.proj == 'CrossAttn':
                shifts_u, shifts_v, theta = \
                    shifts_u, shifts_v, theta  = net.CrossAttn_rot_corr(sat_img, grd_imgs, gt_shift_u, gt_shift_v, gt_heading, mode='test')
            else:
                raise Exception("Sorry, rot_corr not ready yet, use CrossAttn_rot_corr")
                # shifts_u, shifts_v, theta = \
                #     net.rot_corr(sat_map, grd_img, R_FL, T_FL, gt_shift_u, gt_shift_v, gt_heading, mode='test')

            shifts = torch.stack([shifts_u, shifts_v], dim=-1)
            headings = theta.unsqueeze(dim=-1)

            gt_shift = torch.stack([gt_shift_u, gt_shift_v], dim=-1)  # [B, 2]

            pred_shifts.append(shifts.data.cpu().numpy())
            pred_headings.append(headings.data.cpu().numpy())
            gt_shifts.append(gt_shift.data.cpu().numpy())
            gt_headings.append(gt_heading.reshape(-1, 1).data.cpu().numpy())

            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time)/len(testloader)

    pred_shifts = np.concatenate(pred_shifts, axis=0)
    pred_headings = np.concatenate(pred_headings, axis=0) * args.rotation_range
    gt_shifts = np.concatenate(gt_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    gt_headings = np.concatenate(gt_headings, axis=0) * args.rotation_range

    scio.savemat(os.path.join(save_path, str(args.train_log_start) + '_result.mat'), {'gt_shifts': gt_shifts, 'gt_headings': gt_headings,
                                                         'pred_shifts': pred_shifts, 'pred_headings': pred_headings})

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))  # [N]
    angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]

    init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
    init_angle = np.abs(gt_headings)
    diff_shifts = np.abs(pred_shifts - gt_shifts)

    diff_lats = diff_shifts[:, 0]
    diff_lons = diff_shifts[:, 1]

    gt_lats = gt_shifts[:, 0]
    gt_lons = gt_shifts[:, 1]

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, str(args.train_log_start) + '_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    f.write('Time per image (second): ' + str(duration) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print(str(args.train_log_start) + ' Validation results:')

    print('Distance average: (init, pred)', np.mean(init_dis), np.mean(distance))
    print('Distance median: (init, pred)', np.median(init_dis), np.median(distance))

    print('Lateral average: (init, pred)', np.mean(np.abs(gt_lats)), np.mean(diff_lats))
    print('Lateral median: (init, pred)', np.median(np.abs(gt_lats)), np.median(diff_lats))

    print('Longitudinal average: (init, pred)', np.mean(np.abs(gt_lons)), np.mean(diff_lons))
    print('Longitudinal median: (init, pred)', np.median(np.abs(gt_lons)), np.median(diff_lons))

    print('Angle average (init, pred): ', np.mean(np.abs(gt_headings)), np.mean(angle_diff))
    print('Angle median (init, pred): ', np.median(np.abs(gt_headings)), np.median(angle_diff))

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lats < metrics[idx]) / diff_lats.shape[0] * 100
        init = np.sum(np.abs(gt_lats) < metrics[idx]) / gt_lats.shape[0] * 100

        line = 'lateral      within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

        pred = np.sum(diff_lons < metrics[idx]) / diff_lons.shape[0] * 100
        init = np.sum(np.abs(gt_lons) < metrics[idx]) / gt_lons.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()

    net.train()

    return


def train(net, args, save_path, trainloader, testloader):
    accumulation_steps = 4
    base_lr = 1e-4
    optimizer = optim.AdamW(net.parameters(), lr=base_lr)

    net = torch.compile(net)
    torch.set_float32_matmul_precision('high') #no improvement probably because the network is bound not computetionaly but by other factors 
    for epoch in range(args.resume, args.epochs):
        net.train()

        if epoch >= 1:
            base_lr = 1e-5

        optimizer.zero_grad()

        print('batch_size:', args.batch_size, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):
            sat_img, grd_imgs, gt_shift_u, gt_shift_v, gt_heading, grd_names = [[camera.to(args.device) if type(camera) == torch.Tensor else camera for camera in item] if type(item)== list else item.to(args.device) if type(item) == torch.Tensor else item for item in Data]

            t0 = time.time()
            if args.proj == 'CrossAttn':
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                opt_loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                shift_lat_last, shift_lon_last, theta_last, \
                corr_loss = \
                    net.CrossAttn_rot_corr(sat_img, grd_imgs, gt_shift_u, gt_shift_v, gt_heading, mode='train', epoch = epoch)
                    # net.CrossAttn_rot_corr(sat_map, grd_img, R_FL, T_FL, gt_shift_u, gt_shift_v, theta, mode='train', epoch=epoch)
            else:
                raise Exception("Sorry, rot_corr not ready yet, use CrossAttn_rot_corr")
                # opt_loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                # shift_lat_last, shift_lon_last, theta_last, \
                # grd_conf_list, corr_loss = \
                #     net.rot_corr(sat_map, grd_img, R_FL, T_FL, gt_shift_u, gt_shift_v, theta, mode='train', epoch=epoch)

            loss = opt_loss + corr_loss * torch.exp(-net.coe_T) + net.coe_T + net.coe_R
            loss = loss/ accumulation_steps
            loss.backward()
            if (Loop+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
                torch.cuda.synchronize(); t1 = time.time(); print((t1-t0))

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # optimizer.step()
            # optimizer.zero_grad()

            if Loop % 10 == 9:  #
                level = args.level - 1

                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Delta: Level-' + str(level) +
                      ' loss: ' + str(np.round(loss_decrease[level].item(), decimals=4)) +
                      ' lat: ' + str(np.round(shift_lat_decrease[level].item(), decimals=2)) +
                      ' lon: ' + str(np.round(shift_lon_decrease[level].item(), decimals=2)) +
                      ' rot: ' + str(np.round(thetas_decrease[level].item(), decimals=2)))


                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Last: Level-' + str(level) +
                      ' loss: ' + str(np.round(loss_last[level].item(), decimals=4)) +
                      ' lat: ' + str(np.round(shift_lat_last[level].item(), decimals=2)) +
                      ' lon: ' + str(np.round(shift_lon_last[level].item(), decimals=2)) +
                      ' rot: ' + str(np.round(theta_last[level].item(), decimals=2))
                      )


                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                      ' triplet loss: ' + str(np.round(corr_loss.item(), decimals=4)) +
                      ' coe_R: ' + str(np.round(net.coe_R.item(), decimals=2)) +
                      ' coe_T: ' + str(np.round(net.coe_T.item(), decimals=2))
                      )

        print('Save Model ...')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))

        test1(net, args, save_path, testloader, epoch=epoch)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=15, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 1e-2

    parser.add_argument('--rotation_range', type=float, default=10., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    parser.add_argument('--level', type=int, default=3, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--N_iters', type=int, default=2, help='any integer')

    parser.add_argument('--train_log_start', type=int, default=2, help='')
    parser.add_argument('--train_log_end', type=int, default=5, help='')
    parser.add_argument('--test_log_ind', type=int, default=0, help='')

    parser.add_argument('--proj', type=str, default='CrossAttn', help='geo, polar, nn, CrossAttn')

    parser.add_argument('--train_whole', type=int, default=0, help='0 or 1')
    parser.add_argument('--test_whole', type=int, default=0, help='0 or 1')


    parser.add_argument('--use_uncertainty', type=int, default=1, help='0 or 1')
    parser.add_argument('--image_H', type=int, default=256, help='max 860')
    parser.add_argument('--image_W', type=int, default=1024, help='max 1656')
    parser.add_argument('--cameras', type=list, default=['FL', 'SL'], help='max 1656')
    parser.add_argument('--lifting', type=str, default='3D', help = 'homography or 3D lifting')

    args = parser.parse_args(['--batch_size', '1', '--N_iters', '1'])

    return args


def getSavePath(args):
    save_path = './ModelsFord/3DoF/' \
                + 'Log_' + str(args.train_log_start+1) + 'lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range) \
                + '_Nit' + str(args.N_iters) + '_' + str(args.proj) + '_' + '_'.join(args.cameras) + '_' + args.lifting

    if args.use_uncertainty:
        save_path = save_path + '_Uncertainty'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('save_path:', save_path)

    return save_path


# if __name__ == '__main__':
def start_training():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    save_path = getSavePath(args)

    # Train loader 
    train_set = SatGrdDatasetFord(logs=test_logs[args.train_log_start:args.train_log_end],
                                    logs_img_inds=test_logs_img_inds[args.train_log_start:args.train_log_end],
                                    shift_range_lat=args.shift_range_lat,
                                    shift_range_lon=args.shift_range_lon,
                                    rotation_range=args.rotation_range, whole=args.train_whole,
                                    H = args.image_H, W = args.image_W, cameras=args.cameras, mode='train')
    trainloader = DataLoader(train_set, batch_size=mini_batch, shuffle=True, pin_memory=True,
                                num_workers=2, drop_last=False)


    args.Rs = {key:item.to(device) for key, item in train_set.Rs.items()}
    args.Ts = {key:item.to(device) for key, item in train_set.Ts.items()}
    args.Ks = {key:item.to(device) for key, item in train_set.Ks.items()}
    args.device = device

    # test loader 
    test_set = SatGrdDatasetFord(logs=test_logs[args.train_log_start:args.train_log_end],
                                 logs_img_inds=test_logs_img_inds[args.train_log_start:args.train_log_end],
                                  shift_range_lat=args.shift_range_lat, shift_range_lon=args.shift_range_lon,
                                  rotation_range=args.rotation_range, whole=args.test_whole, 
                                  H = args.image_H, W = args.image_W, cameras=args.cameras, mode='test')
    testloader = DataLoader(test_set, batch_size=mini_batch, shuffle=False, pin_memory=True,
                             num_workers=2, drop_last=False)

    net = ModelFord(args)
    net.to(device)


    net.load_state_dict(torch.load(os.path.join('ModelsFord/3DoF/Log_1lat20.0m_lon20.0m_rot10.0_Nit1_CrossAttn_FL_SL_3D_Uncertainty', 'model_7.pth')), strict=False)

    if args.test:
        net.load_state_dict(torch.load(os.path.join(save_path, 'model_4.pth')), strict=False)
        test1(net, args, save_path, testloader)

    else:
        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')))
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')


        train(net, args, save_path, trainloader, testloader)

