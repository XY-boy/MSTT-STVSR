import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import models.pyflow as pyflow
from skimage import img_as_float
from random import randrange
import os.path

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]

        for i in seq:
            index = int(filepath[char_len-7:char_len-4])-i
            file_name=filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png'
            
            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png').convert('RGB'),scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
    
    return target, input, neigbor

def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames/2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        if nFrames%2 == 0:
            seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt,tt+1) if x!=0]
        #random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len-7:char_len-4])+i
            file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.png'
            
            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp=input
                neigbor.append(temp)
            
    else:
        seqGT = [1,2,3,4,5,6,7]
        seqLQs = [1,3,5,7]
        target = []
        input = []
        for j in seqGT:
            target.append(modcrop(Image.open(filepath + '/im'+str(j)+'.png').convert('RGB'),scale))
        for i in seqLQs:
            input.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target[0].size[0]/scale),int(target[0].size[1]/scale)),Image.BICUBIC))
        # target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
            #         # input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
            #         # neigbor = []
            #         # seq = [x for x in range(4-tt,5+tt) if x!=4]
            #         # #random.shuffle(seq) #if random sequence
            #         # for j in seq:
        #     neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow

def rescale_flow(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in[0].size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    # img_in = img_in.crop((iy,ix,iy + ip, ix + ip))#[:, iy:iy + ip, ix:ix + ip]
    img_in = [i.crop((iy,ix,iy + ip, ix + ip)) for i in img_in]#[:, iy:iy + ip, ix:ix + ip]
    # img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))#[:, ty:ty + tp, tx:tx + tp]
    img_tar = [j.crop((ty,tx,ty + tp, tx + tp)) for j in img_tar]#[:, ty:ty + tp, tx:tx + tp]
    # img_nn = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_nn] #[:, iy:iy + ip, ix:ix + ip]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch

def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = [ImageOps.flip(j) for j in img_in]
        # img_in = ImageOps.flip(img_in)
        img_tar = [ImageOps.flip(i) for i in img_tar]
        # img_tar = ImageOps.flip(img_tar)
        # img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = [ImageOps.mirror(j) for j in img_in]
            img_tar = [ImageOps.mirror(i) for i in img_tar]
            # img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = [j.rotate(180) for j in img_in]
            img_tar = [i.rotate(180) for i in img_tar]
            # img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, info_aug
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, transform=None):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)

        if self.patch_size != 0:
            input, target, _ = get_patch(input,target,self.patch_size, self.upscale_factor, self.nFrames)
        
        if self.data_augmentation:
            input, target, _ = augment(input, target)

        f_flow = []
        b_flow = []
        for i in range(len(input) - 1):
            f_flow.append(get_flow(input[i], input[i+1]))
            b_flow.append(get_flow(input[i+1], input[i]))
        # print(f_flow[0].shape)
            
        # # flow = [get_flow(input,j) for j in neigbor]


            
        # bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            target = [self.transform(j) for j in target]
            input = [self.transform(i) for i in input]
            # bicubic = self.transform(bicubic)
            # neigbor = [self.transform(j) for j in neigbor]
            # fw_frame = [torch.from_numpy(j.transpose(2,0,1)) for j in fw_frame]
            # bw_frame = [torch.from_numpy(j.transpose(2,0,1)) for j in bw_frame]

        return input, target, f_flow, b_flow

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
            
        flow = [get_flow(input,j) for j in neigbor]

        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
            
        return input, target, neigbor, flow, bicubic
      
    def __len__(self):
        return len(self.image_filenames)
