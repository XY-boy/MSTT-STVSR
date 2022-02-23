import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_networks import *
# from dbpns import Net as DBPNS


import torchvision.models as models

# from core.spectral_norm import spectral_norm as _spectral_norm

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class MSTT(BaseNetwork):
    def __init__(self, init_weights=True, n_frame=7, stack_nums=1):
        super(MSTT, self).__init__()
        channel = 256
        nf = 64

        # stack_num = 8
        self.stack_num = stack_nums
        self.n_frame = n_frame
        # NON-OVERLAPPING
        patchsize = [(40, 40), (20, 20), (10, 10), (5, 5)]
        blocks = []

        for _ in range(self.stack_num):
            blocks.append(TransformerBlock(patchsize, hidden=channel))
        self.transformer = nn.Sequential(*blocks)

        # encoder: encode features
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 1/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoder: decode features
        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),  # *2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, lrs):
        # extracting features
        # B, N, C, H, W = lrs.size()  # B * 7 * 64 * patchsize * patchsize(ps)

        b, t, c, h, w = lrs.size()
        enc_feat = self.encoder(lrs.view(b*t, c, h, w))
        _, c, h, w = enc_feat.size()
        enc_feat = self.transformer(
            {'x': enc_feat, 'b': b, 'c': c})['x']  # Transformer，MHA，[b*t,256,h/4,w/4]
        output = self.decoder(enc_feat)  # 【b*t，64，h，w】
        output = torch.tanh(output)  # [b*t,64,160,160]

        output = lrs + output.view(b, t, 64, h*2, w*2)

        return output


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)

# #############################################################################
# ############################# Transformer  ##################################
#  Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
# #############################################################################

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))  # [1,80,80]
        # scores_2 = torch.sum(query * key, -1) / math.sqrt(64)
        # print(scores_2.size())
        p_attn = F.softmax(scores, dim=-1)  # [1,80,80]
        p_val = torch.matmul(p_attn, value)  # [1, 80, 6400]

        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):  # d_model = 256
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, b, c):
        bt, _, h, w = x.size()  # x.size()=[b*t,256,h/4,w/4]
        t = bt // b  # 帧数
        d_k = c // len(self.patchsize)  # len(patchsize)为head的个数，4个，d_k = 256/4=64
        output = []
        _query = self.query_embedding(x)  # 通过卷积将encode的特征嵌入到查询特征query=【b*t,256,h/4,w/4】
        _key = self.key_embedding(x)  # 同理，key = [b*t,256,h/4,w/4]
        _value = self.value_embedding(x)  # _value = [b*t,256,h/4,w/4]

        #  torch.chunk,与cat拼接相反，分割操作,每个head分到256/4=64维度的特征
        # 假设HR=640*640，LR=160*160，经过encoder之后，变为40*40，patchsize可以设为10*10，这样就能得到：
        # 40*40*5（5帧）/10*10=80个不重叠的patch，与原文的t*(h/r1)*(w/r2)一致
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height  # 40/patch=40/10=4

            # 1) embedding and reshape
            # query原本是[1,5,64,40,40],通道上被分成了4块
            query = query.view(b, t, d_k, out_h, height, out_w, width)  # [1,5,64,4,10,4,10]
            # query.permute(0, 1, 3, 5, 2, 4, 6).size()=[1,5,4,4,64,10,10]
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)  # [1,80,6400]
            key = key.view(b, t, d_k, out_h, height, out_w, width)  # [1,5,64,4,10,4,10]
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)  # [1,80,6400]
            value = value.view(b, t, d_k, out_h, height, out_w, width)  # [1,5,64,4,10,4,10]
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)   # [1,80,6400]
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.attention(query, key, value)  # 返回注意力调制后的value赋给y，注意力赋给_
            # 3) "Concat" using a view and apply a final linear.
            # y.size()=[1,80,6400]
            y = y.view(b, t, out_h, out_w, d_k, height, width)  # [1,5,4,4,64,10,10]
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)  # [5,64,40,40]
            output.append(y)
        # output存储每个head的结果，有四个结果，每个结果都是[5,64,40,40]
        output = torch.cat(output, 1)  # [5,256,40,40]
        output = self.output_linear(output)  # [5,256,40,40]
        return x + output


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, b, c = x['x'], x['b'], x['c']
        # print(x.size())
        x = x + self.attention(x, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'b': b, 'c': c}


# ######################################################################
# ######################################################################


# def spectral_norm(module, mode=True):
#     if mode:
#         return _spectral_norm(module)
#     return module

class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)

        if self.activation is not None:
            out = self.act(out)

        return out

class ConvBlock(torch.nn.Module):  # 不带BN的Conv
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out