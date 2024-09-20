import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

import torch.optim as optim

from dataLoader.Ford_dataset import SatGrdDatasetFord, SatGrdDatasetFordTest, train_logs, train_logs_img_inds, test_logs, test_logs_img_inds

import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

from models_ford import ModelFord

import numpy as np
import os
import argparse

from torch.utils.data import DataLoader
import time

torch.autograd.set_detect_anomaly(True)

def test1(net, args, save_path, test_log_ind=1, epoch=0, device=torch.device("cuda:0")):

    net.eval()
    mini_batch = args.batch_size

    np.random.seed(2022)
    torch.manual_seed(2022)

    test_set = SatGrdDatasetFordTest(logs=test_logs[test_log_ind:test_log_ind+1],
                                     logs_img_inds=test_logs_img_inds[test_log_ind:test_log_ind+1],
                                     shift_range_lat=args.shift_range_lat,
                                     shift_range_lon=args.shift_range_lon,
                                     rotation_range=args.rotation_range, whole=args.test_whole)
    testloader = DataLoader(test_set, batch_size=mini_batch, shuffle=False, pin_memory=True,
                            num_workers=2, drop_last=False)

    pred_shifts = []
    gt_shifts = []

    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            sat_map, grd_img, gt_shift_u, gt_shift_v, gt_heading, R_FL, T_FL = \
                [item.to(device) for item in data[:-1]]

            if args.proj == 'CrossAttn':
                shifts_u, shifts_v = \
                    net.CrossAttn_corr(sat_map, grd_img, R_FL, T_FL, gt_shift_u, gt_shift_v, torch.zeros_like(gt_heading), mode='test')

            shifts = torch.stack([shifts_u, shifts_v], dim=-1)
            gt_shift = torch.stack([gt_shift_u, gt_shift_v], dim=-1)  # [B, 2]

            pred_shifts.append(shifts.data.cpu().numpy())
            gt_shifts.append(gt_shift.data.cpu().numpy())

            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time)/len(testloader)

    pred_shifts = np.concatenate(pred_shifts, axis=0)
    gt_shifts = np.concatenate(gt_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    scio.savemat(os.path.join(save_path, str(test_log_ind) + '_result.mat'), {'gt_shifts': gt_shifts, 'pred_shifts': pred_shifts})

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))  # [N]

    init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
    diff_shifts = np.abs(pred_shifts - gt_shifts)

    diff_lats = diff_shifts[:, 0]
    diff_lons = diff_shifts[:, 1]

    gt_lats = gt_shifts[:, 0]
    gt_lons = gt_shifts[:, 1]

    metrics = [1, 3, 5]

    f = open(os.path.join(save_path, str(test_log_ind) + '_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    f.write('Time per image (second): ' + str(duration) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print(str(test_log_ind) + ' Validation results:')

    print('Distance average: (init, pred)', np.mean(init_dis), np.mean(distance))
    print('Distance median: (init, pred)', np.median(init_dis), np.median(distance))

    print('Lateral average: (init, pred)', np.mean(np.abs(gt_lats)), np.mean(diff_lats))
    print('Lateral median: (init, pred)', np.median(np.abs(gt_lats)), np.median(diff_lats))

    print('Longitudinal average: (init, pred)', np.mean(np.abs(gt_lons)), np.mean(diff_lons))
    print('Longitudinal median: (init, pred)', np.median(np.abs(gt_lons)), np.median(diff_lons))


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

    print('====================================')
    f.write('====================================\n')
    f.close()

    net.train()
    return


def train(args, save_path, train_log_start=1, train_log_end=2):

    for epoch in range(args.resume, args.epochs):
        net.train()

        base_lr = 1e-4
        if epoch >= 2:
            base_lr = 1e-5

        optimizer = optim.Adam(net.parameters(), lr=base_lr)
        optimizer.zero_grad()

        train_set = SatGrdDatasetFord(logs=train_logs[train_log_start:train_log_end],
                                      logs_img_inds=train_logs_img_inds[train_log_start:train_log_end],
                                      shift_range_lat=args.shift_range_lat,
                                      shift_range_lon=args.shift_range_lon,
                                      rotation_range=args.rotation_range,
                                      whole=args.train_whole)
        trainloader = DataLoader(train_set, batch_size=mini_batch, shuffle=True, pin_memory=True,
                                 num_workers=1, drop_last=False)
        print('batch_size:', mini_batch, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):
            sat_map, grd_img, gt_shift_u, gt_shift_v, theta, R_FL, T_FL = \
                [item.to(device) for item in Data[:-1]]

            optimizer.zero_grad()

            if args.proj == 'CrossAttn':
                loss = net.CrossAttn_corr(sat_map, grd_img, R_FL, T_FL, gt_shift_u, gt_shift_v, theta, mode='train', epoch=epoch)

            loss.backward()
            optimizer.step()  # This step is responsible for updating weights
            optimizer.zero_grad()

            if Loop % 10 == 9:  #

                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                      ' triplet loss: ' + str(np.round(loss.item(), decimals=4))
                      )

        print('Save Model ...')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))
        test1(net, args, save_path, test_log_ind=train_log_start, epoch=epoch)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=1, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--rotation_range', type=float, default=0., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('--level', type=int, default=3, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--N_iters', type=int, default=2, help='any integer')

    parser.add_argument('--Optimizer', type=str, default='TransV1G2SP', help='it does not matter in the orientation-aligned setting')

    parser.add_argument('--train_log_start', type=int, default=0, help='')
    parser.add_argument('--train_log_end', type=int, default=1, help='')
    parser.add_argument('--test_log_ind', type=int, default=0, help='')

    parser.add_argument('--proj', type=str, default='CrossAttn', help='geo, CrossAttn')

    parser.add_argument('--train_whole', type=int, default=0, help='0 or 1')
    parser.add_argument('--test_whole', type=int, default=0, help='0 or 1')

    parser.add_argument('--use_uncertainty', type=int, default=1, help='0 or 1')

    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path = './ModelsFord/2DoF/' \
                + 'Log_' + str(args.train_log_start+1) + 'lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm' \
                + '_' + str(args.proj)

    if args.use_uncertainty:
        save_path = save_path + '_Uncertainty'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('save_path:', save_path)

    return save_path



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    save_path = getSavePath(args)

    net = ModelFord(args)
    net.to(device)

    if args.test:
        net.load_state_dict(torch.load(os.path.join(save_path, 'model_4.pth')), strict=False)
        test1(net, args, save_path, args.train_log_start, device=device)

    else:

        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')))
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')

        train(args, save_path, train_log_start=args.train_log_start, train_log_end=args.train_log_end)


