''' network architecture for Sakuya '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.module_util as mutil
from models.convlstm import ConvLSTM, ConvLSTMCell
from models.base_networks import *
from models.Spatial_Temporal_Transformer import MSTT as MSST_former
try:
    from models.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class MSTTr(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10):
        super(MSTTr, self).__init__()
        self.nf = nf
        self.in_frames = 1 + nframes // 2
        self.ot_frames = nframes
        p_size = 48 # a place holder, not so useful
        patch_size = (p_size, p_size) 
        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(nf)

        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, front_RBs)

        # MSD alignment module proposed in 10.1109/TGRS.2021.3107352
        self.MSD_align = my_align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)  # blending
        self.blending = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)  # final blending

        self.MSST_former = MSST_former(True, n_frame=7, stack_nums=1)

        #### reconstruction
        self.recon_trunk = mutil.make_layer(ResidualBlock_noBN_f, back_RBs)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, f_flow, b_flow):
        B, N, C, H, W = x.size()  # N input video frames

        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)  # [4,64,h,w]
        L1_fea = L1_fea.view(B, N, -1, H, W)  # 1 3 5 7 frame

        MSD_f_fea = []
        MSD_b_fea = []
        flow_f_fea = []
        flow_b_fea = []
        blending1 = []
        blending2 = []
        blending_final = []

        for i in range(N -1):  # 0 1 2
            MSD_f_fea.append(self.MSD_align(L1_fea[:,i,:,:,:], L1_fea[:,i+1,:,:,:]))  # F2+，F4+，F6+
            flow_b_fea.append(mutil.flow_warp(L1_fea[:,i+1,:,:,:], f_flow[i]))  # F2-，F4-，F6-
            blending1.append(self.fusion(torch.cat([MSD_f_fea[i], flow_b_fea[i]],dim=1)))  # fusion

            MSD_b_fea.append(self.MSD_align(L1_fea[:,i+1,:,:,:], L1_fea[:,i,:,:,:]))  # F2-，F4-，F6-
            flow_f_fea.append(mutil.flow_warp(L1_fea[:,i,:,:,:], b_flow[i]))  # F2+，F4+，F6+
            blending2.append(self.fusion(torch.cat([flow_f_fea[i], flow_b_fea[i]],dim=1)))  # fusion

            blending_final.append(self.blending(torch.cat([blending1[i], blending2[i]], dim=1)))  # F2 F4 F6

        to_mstt_fea = torch.stack([L1_fea[:,0,:,:,:], blending_final[0], L1_fea[:,1,:,:,:], blending_final[1],L1_fea[:,2,:,:,:],blending_final[2],L1_fea[:,3,:,:,:]], dim=1)

        feats = self.MSST_former(to_mstt_fea)
        B, T, C, H, W = feats.size()

        feats = feats.view(B*T, C, H, W)
        out = self.recon_trunk(feats)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        _, _, K, G = out.size()
        outs = out.view(B, T, -1, K, G)
        return outs