import torch
import os
import time
import socket
import glob
import os.path as osp
import numpy as np
import cv2
import torch.nn as nn

from torch.utils.data import DataLoader

from data.data import get_training_set, get_eval_set


import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter  # tensorboard log
import utils.util as util

import argparse
import math
import torch.backends.cudnn as cudnn


from models.Model_arch import MSTTr as STVSR

from loss import CharbonnierLoss as Cb_Loss
import options.options as option
import models.pyflow as pyflow
from thop import profile

# -- Parameter Setting --
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='./options/train/train_zsm.yml', help='Path to option YAML file.')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')

parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=0.01')
parser.add_argument('--data_dir', type=str, default='D:\Dataset\VSR/189_vime7/train')
parser.add_argument('--file_list', type=str, default='189_vime7_train.txt')
parser.add_argument('--patch_size', type=int, default=80, help='0 to use original frame size')

# -- Test while Training ———
parser.add_argument('--test_dir', type=str, default='./test_example/*')

parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--pretrained_sr', default='4x_DESKTOP-0NFK80A_epoch_25.pth',
                    help='sr pretrained base model default=4x_DESKTOP-0NFK80A_epoch_25.pth')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=True)
cudnn.benchmark = True
gpus_list = range(1)  # number of GPUs

hostname = str(socket.gethostname())
if torch.cuda.is_available():
    device = torch.device('cuda')

writer = SummaryWriter('runs/MSTT_STVSR')

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: {:03f} M'.format((num_params / 1e6)))

def checkpoint(epoch):
    model_out_path = args.save_folder + str('4x_' + hostname + "_epoch_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def save_best_model(bestepoch):
    model_out_path = args.save_folder + 'best_' + str('4x_' + "_epoch_{}.pth".format(bestepoch))
    torch.save(model.state_dict(), model_out_path)
    print("BestModel saved to {}".format(model_out_path))

def single_forward(model, imgs_in):  # used for test while training
    with torch.no_grad():
        # imgs_in.size(): [1,n,3,h,w] tensor
        b,n,c,h,w = imgs_in.size()
        # resize for scale = 4
        h_n = int(4*np.ceil(h/4))
        w_n = int(4*np.ceil(w/4))

        imgs_temp = imgs_in.new_zeros(b,n,c,h_n,w_n)
        imgs_temp[:,:,:,0:h,0:w] = imgs_in.cuda(0)

        # flow calucation
        forward_flow = []
        backward_flow = []
        for i in range(n - 1):
            im1 = np.array(imgs_in[:, i, :, :, :].cpu()).astype(float)
            im2 = np.array(imgs_in[:, i + 1, :, :, :].cpu()).astype(float)

            im1 = np.squeeze(im1, 0).transpose(1, 2, 0).copy()
            im2 = np.squeeze(im2, 0).transpose(1, 2, 0).copy()

            # Flow Options:
            alpha = 0.012
            ratio = 0.75
            minWidth = 20
            nOuterFPIterations = 7
            nInnerFPIterations = 1
            nSORIterations = 30
            colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

            u, v, im2W = pyflow.coarse2fine_flow(
                im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                nSORIterations, colType)
            u2, v2, im2W2 = pyflow.coarse2fine_flow(
                im2, im1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                nSORIterations, colType)
            f_flow = np.concatenate((u[..., None], v[..., None]), axis=2)  # forward flow
            b_flow = np.concatenate((u2[..., None], v2[..., None]), axis=2)  # backward flow

            # h*w*2-->2*h*w
            f_flow = torch.from_numpy(f_flow).float()  # [2 32 32]
            b_flow = torch.from_numpy(b_flow).float()  # [2 32 32]
            # 2*h*w-->1*2*h*w
            f_flow = torch.unsqueeze(f_flow, dim=0).cuda(0)  # [1 2 32 32]
            b_flow = torch.unsqueeze(b_flow, dim=0).cuda(0)  # [1 2 32 32]

            forward_flow.append(f_flow)
            backward_flow.append(b_flow)

        model_output = model(imgs_temp, forward_flow, backward_flow)
        # model_output.size(): torch.Size([1, 3, 4h, 4w])
        model_output = model_output[:, :, :, 0:4*h, 0:4*w]
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    return output

# --No distributed training --
train_sampler = None
opt['dist'] = False

# -- Random seed setting --
seed = opt['train']['manual_seed']
torch.manual_seed(seed)
if opt['gpu_ids']:
    torch.cuda.manual_seed(seed)

# -- Load model --
print('==>Load model.')
model = STVSR(nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=20)
print_network(model)

# -- Load checkpoint--
if args.pretrained:
    model_name = os.path.join(args.save_folder + args.pretrained_sr)
    # print(model_name)
    if os.path.exists(model_name):
        # model = torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage), strict=False)
        print('Pre-trained SR model is loaded.')

# -- loss function --
if opt['gpu_ids']:
    model.cuda()
    criterion = Cb_Loss().cuda()
    # criterion = nn.L1Loss().cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

# -- load training data --
train_set = get_training_set(args.data_dir, 7, 4, True, args.file_list, False, args.patch_size, True)
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batchSize, shuffle=True)

# -- begin training --
best_epoch = 0
best_test_psnr = 0.0
for epoch in range(args.start_epoch, args.nEpochs + 1):
    epoch_loss = 0
    current_iter = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, gt, f_flow, b_flow = batch[0], batch[1], batch[2], batch[3]

        gt = torch.stack([Variable(j).cuda(gpus_list[0]) for j in gt], dim=1)  # [B 7 3 H W]
        input = torch.stack([Variable(i).cuda(gpus_list[0]) for i in input],dim=1)  # [B 4 3 h w]
        f_flow = [Variable(j).cuda(gpus_list[0]).float() for j in f_flow]
        b_flow = [Variable(j).cuda(gpus_list[0]).float() for j in b_flow]

        optimizer.zero_grad()

        t0 = time.time()
        prediction = model(input, f_flow, b_flow)
        t1 = time.time()

        loss = criterion(prediction, gt)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(training_data_loader),
                                                                                 loss.item(),
                                                                                 (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    writer.add_scalar('Avg. Loss', epoch_loss / len(training_data_loader), epoch)

    if (epoch+1) % (args.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    checkpoint(epoch)

    # test
    psnr_predicted = 0.0
    avg_test_psnr = 0.0
    sub_folder_name_l = []

    model.eval()
    sub_folder_l = sorted(glob.glob(args.test_dir))
    gt_tested_list = []
    sub_folder_name = args.test_dir.split('/')[-1]
    sub_folder_name_l.append(sub_folder_name)
    sub_folder = args.test_dir + '/input4/'
    img_LR_l = sorted(glob.glob(sub_folder + '/*'))
    save_sub_folder = './test_example3/temp_results' + '/' + str(epoch)
    util.mkdir(save_sub_folder)
    print(save_sub_folder)

    #### read LR images
    imgs = util.read_seq_imgs(sub_folder)  # read images to numpy
    #### read GT images
    img_GT_l = []
    sub_folder_GT = osp.join(sub_folder.replace('/input4/', '/truth/'))
    for im_GT_path in sorted(glob.glob(osp.join(sub_folder_GT, '*'))):
        img_GT_l.append(util.read_image(im_GT_path))
    select_idx_list = util.test_index_generation(True, 7, len(img_LR_l))
    # print(img_GT_l)

    for select_idxs in select_idx_list:
        # get input images
        select_idx = select_idxs[0]  # LRs
        gt_idx = select_idxs[1]  # GT
        # print(gt_idx)
        imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)  # [1,4,3,h,w]ou
        output = single_forward(model, imgs_in)
        outputs = output.data.float().cpu().squeeze(0)

        for idx, name_idx in enumerate(gt_idx):
            if name_idx in gt_tested_list:
                continue
            gt_tested_list.append(name_idx)
            output_f = outputs[idx, :, :, :].squeeze(0)

            output = util.tensor2img(output_f)

            cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(name_idx + 1)), output)
            # PSNR
            output = output / 255.
            # print(name_idx)
            GT = np.copy(img_GT_l[name_idx])
            cropped_output = output
            cropped_GT = GT
            crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
            psnr_predicted += crt_psnr
    avg_test_psnr = psnr_predicted / 7
    print('Test avg.PSNR = {:.4f} dB'.format(avg_test_psnr))
    writer.add_scalar('Avg. psnr', avg_test_psnr, epoch)