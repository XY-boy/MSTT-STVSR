'''
test Zooming Slow-Mo models on arbitrary datasets
write to txt log file
[kosame] TODO: update the test script to the newest version
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.Sakuya_arch as Sakuya_arch
import models.pyflow as pyflow
import time

def main():
    scale = 4
    N_ot = 7 #3
    N_in = 1+ N_ot // 2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #### model 
    #### TODO: change your model path here
    model_path = './weights/4x_DESKTOP-MFC5BAN_epoch_48.pth'
    model = Sakuya_arch.LunaTokis(64, N_ot, 8, 5, 40)
    # model = torch.nn.DataParallel(model)


    #### dataset
    data_mode = 'SPMC' #'Vid4' #'SPMC'#'Middlebury'#

    if data_mode == 'Vid4':
        test_dataset_folder = '/data/xiang/SR/Vid4/LR/*'
    if data_mode == 'SPMC':
        test_dataset_folder = 'D:/Github-package/ZoomingSM-jilin189/jilin_test12/*'
    if data_mode == 'Custom':
        test_dataset_folder = '../test_example/*' # TODO: put your own data path here

    #### evaluation
    flip_test = False #True#
    crop_border = 0

    # temporal padding mode
    padding = 'replicate'
    save_imgs = True #True#
    if 'Custom' in data_mode: save_imgs = True
    ############################################################################
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')
    save_folder = './results_ours/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    model_params = util.get_model_total_params(model)

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Model parameters: {} M'.format(model_params))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))

    def single_forward(model, imgs_in):
        with torch.no_grad():
            # imgs_in.size(): [1,n,3,h,w] tensor
            b, n, c, h, w = imgs_in.size()
            # print(imgs_in.size())
            h_n = int(4 * np.ceil(h / 4))
            w_n = int(4 * np.ceil(w / 4))
            # h_n = h
            # w_n = w
            imgs_temp = imgs_in.new_zeros(b, n, c, h_n, w_n)
            imgs_temp[:, :, :, 0:h, 0:w] = imgs_in.cuda(0)


            forward_flow = []
            backward_flow = []
            for i in range(n - 1):
                im1 = np.array(imgs_in[:, i, :, :, :].cpu()).astype(float)  # (1,3,64,112)
                im2 = np.array(imgs_in[:, i + 1, :, :, :].cpu()).astype(float)

                im1 = np.squeeze(im1, 0).transpose(1, 2, 0).copy()  # (64,112,3)
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
                f_flow = np.concatenate((u[..., None], v[..., None]), axis=2)
                b_flow = np.concatenate((u2[..., None], v2[..., None]), axis=2)
                # h*w*2-->2*h*w
                f_flow = torch.from_numpy(f_flow).float()  # [2 32 32]
                b_flow = torch.from_numpy(b_flow).float()  # [2 32 32]
                # 2*h*w-->1*2*h*w
                f_flow = torch.unsqueeze(f_flow, dim=0).cuda(0)  # [1 2 32 32]
                b_flow = torch.unsqueeze(b_flow, dim=0).cuda(0)  # [1 2 32 32]
                forward_flow.append(f_flow)
                backward_flow.append(b_flow)

            t0 = time.time()
            model_output = model(imgs_temp, forward_flow, backward_flow)
            t1 = time.time()
            print(t1-t0)
            # model_output.size(): torch.Size([1, 3, 4h, 4w])
            model_output = model_output[:, :, :, 0:4 * h, 0:4 * w]
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output

    sub_folder_l = sorted(glob.glob(test_dataset_folder))

    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_psnr_y_l = []
    sub_folder_name_l = []
    # total_time = []
    # for each sub-folder
    for sub_folder in sub_folder_l:
        gt_tested_list = []
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)

        if data_mode == 'SPMC':
            sub_folder = sub_folder + '/input4/'
        img_LR_l = sorted(glob.glob(sub_folder + '/*'))
        print(save_sub_folder)

        if save_imgs:
            util.mkdirs(save_sub_folder)

        #### read LR images
        imgs = util.read_seq_imgs(sub_folder)
        #### read GT images
        img_GT_l = []
        if data_mode == 'SPMC':
            sub_folder_GT = osp.join(sub_folder.replace('/input4/', '/truth/'))
        else:
            sub_folder_GT = osp.join(sub_folder.replace('/LR/', '/HR/'))

        if 'Custom' not in data_mode:
            for img_GT_path in sorted(glob.glob(osp.join(sub_folder_GT,'*'))):
                img_GT_l.append(util.read_image(img_GT_path))

        avg_psnr, avg_psnr_sum, cal_n = 0,0,0
        avg_psnr_y, avg_psnr_sum_y = 0,0
        
        if len(img_LR_l) == len(img_GT_l):
            skip = True
        else:
            skip = False
        
        if 'Custom' in data_mode:
            select_idx_list = util.test_index_generation(False, N_ot, len(img_LR_l))
        else:
            select_idx_list = util.test_index_generation(skip, N_ot, len(img_LR_l))
        # process each image
        for select_idxs in select_idx_list:
            # get input images
            select_idx = select_idxs[0]
            gt_idx = select_idxs[1]
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            t1 = time.time()
            output = single_forward(model, imgs_in)
            t0 = time.time()
            # print(t0-t1)
            outputs = output.data.float().cpu().squeeze(0)            

            if flip_test:
                # flip W
                output = single_forward(model, torch.flip(imgs_in, (-1, )))
                output = torch.flip(output, (-1, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip H
                output = single_forward(model, torch.flip(imgs_in, (-2, )))
                output = torch.flip(output, (-2, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip both H and W
                output = single_forward(model, torch.flip(imgs_in, (-2, -1)))
                output = torch.flip(output, (-2, -1))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output

                outputs = outputs / 4

            # save imgs
            for idx, name_idx in enumerate(gt_idx):
                if name_idx in gt_tested_list:
                    continue
                gt_tested_list.append(name_idx)
                output_f = outputs[idx,:,:,:].squeeze(0)

                output = util.tensor2img(output_f)
                if save_imgs:                
                    cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(name_idx+1)), output)

                if 'Custom' not in data_mode:
                    #### calculate PSNR
                    output = output / 255.

                    GT = np.copy(img_GT_l[name_idx])

                    if crop_border == 0:
                        cropped_output = output
                        cropped_GT = GT
                    else:
                        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
                        cropped_GT = GT[crop_border:-crop_border, crop_border:-crop_border, :]
                    crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
                    cropped_GT_y = data_util.bgr2ycbcr(cropped_GT, only_y=True)
                    cropped_output_y = data_util.bgr2ycbcr(cropped_output, only_y=True)
                    crt_psnr_y = util.calculate_psnr(cropped_output_y * 255, cropped_GT_y * 255)
                    logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB  PSNR-Y: {:.6f} dB'.format(name_idx + 1, name_idx+1, crt_psnr, crt_psnr_y))
                    avg_psnr_sum += crt_psnr
                    avg_psnr_sum_y += crt_psnr_y
                    cal_n += 1

        if 'Custom' not in data_mode:
            avg_psnr = avg_psnr_sum / cal_n
            avg_psnr_y = avg_psnr_sum_y / cal_n
    
            logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} frames; '.format(sub_folder_name, avg_psnr, avg_psnr_y, cal_n))
    
            avg_psnr_l.append(avg_psnr)
            avg_psnr_y_l.append(avg_psnr_y)

    if 'Custom' not in data_mode:
        logger.info('################ Tidy Outputs ################')
        for name, psnr, psnr_y in zip(sub_folder_name_l, avg_psnr_l, avg_psnr_y_l):
            logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB. '
                       .format(name, psnr, psnr_y))
        logger.info('################ Final Results ################')
        logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
        logger.info('Padding mode: {}'.format(padding))
        logger.info('Model path: {}'.format(model_path))
        logger.info('Save images: {}'.format(save_imgs))
        logger.info('Flip Test: {}'.format(flip_test))
        logger.info('Total Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} clips. '
                    .format(
                        sum(avg_psnr_l) / len(avg_psnr_l), sum(avg_psnr_y_l) / len(avg_psnr_y_l), len(sub_folder_l)))
        # logger.info('Total Runtime: {:.6f} s Average Runtime: {:.6f} for {} images.'
                    # .format(sum(total_time), sum(total_time)/171, 171))

if __name__ == '__main__':
    main()