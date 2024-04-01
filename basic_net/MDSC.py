from typing import Any, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch import autograd
# import models.basicblock as B
import torch.nn.functional as F
import numpy as np
from math import ceil
import math
from ptflops import get_model_complexity_info
# from .utils import *
__all__ = ['MMCSC', 'cu_mmcsc_n1_d8', 'cu_mmcsc_n1_d16', 'cu_mmcsc_n1_d32','cu_mmcsc_n3_d8','cu_mmcsc_n2_d16','cu_mmcsc_n4_d16','cu_mmcsc_n5_d32']
# iteration version with beta
class Coarse_CSC_layer(nn.Module):
    def __init__(self, num_iter, in_channels, num_filters, kernel_size,stride):
        super(Coarse_CSC_layer, self).__init__()
        self.num_iter = num_iter
        self.in_channel = in_channels
        self.kernel_size = kernel_size
        self.padding=(self.kernel_size-1)//2
        self.num_filters = num_filters
        self.stride = stride

        self.down_conv = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, bias=False)
        self.up_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, bias=False)
        self.lam = nn.Parameter(0.01 * torch.ones(1, self.num_filters, 1, 1))
        self.restore_conv = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, bias=False)
        nn.init.xavier_uniform_(self.restore_conv.weight.data)
        nn.init.xavier_uniform_(self.down_conv.weight.data)
        nn.init.xavier_uniform_(self.up_conv.weight.data)

    def forward(self, mod):
        p1 = self.up_conv(mod)
        tensor = torch.mul(torch.sign(p1), F.relu(torch.abs(p1) - self.lam))
        for i in range(self.num_iter):
            p3 = self.down_conv(tensor)
            p4 = self.up_conv(p3)
            p5 = tensor - p4
            p6 = torch.add(p1, p5)
            tensor = torch.mul(torch.sign(p6), F.relu(torch.abs(p6) - self.lam))
        restore = self.restore_conv(tensor)
        return tensor, restore
class Fine_CSC_layer(nn.Module):
    def __init__(self, num_iter, in_channels, num_filters, kernel_size,stride):
        super(Fine_CSC_layer, self).__init__()
        self.num_iter = num_iter
        self.in_channel = in_channels
        self.kernel_size = kernel_size
        self.padding=(self.kernel_size-1)//2
        self.num_filters = num_filters
        self.stride = stride

        self.down_conv = nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.in_channel,kernel_size=self.kernel_size, padding=self.padding, output_padding=self.stride-self.kernel_size+2*self.padding, stride=self.stride, bias=False)
        self.up_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, bias=False)
        self.lam = nn.Parameter(0.01 * torch.ones(1, self.num_filters, 1, 1))
        self.restore_conv = nn.ConvTranspose2d(in_channels=self.num_filters, out_channels=self.in_channel, kernel_size=self.kernel_size, padding=self.padding, output_padding=self.stride - self.kernel_size + 2 * self.padding, stride=self.stride, bias=False)
        nn.init.xavier_uniform_(self.restore_conv.weight.data)
        nn.init.xavier_uniform_(self.down_conv.weight.data)
        nn.init.xavier_uniform_(self.up_conv.weight.data)
    def forward(self, mod):
        p1 = self.up_conv(mod)
        tensor = torch.mul(torch.sign(p1), F.relu(torch.abs(p1) - self.lam))
        for i in range(self.num_iter):
            p3 = self.down_conv(tensor)
            p4 = self.up_conv(p3)
            p5 = tensor - p4
            p6 = torch.add(p1, p5)
            tensor = torch.mul(torch.sign(p6), F.relu(torch.abs(p6) - self.lam))
        restore = self.restore_conv(tensor)
        return tensor, restore
class Coarse_CU_Encoder(nn.Module):
    def __init__(self,num_iter,in_channel,num_filters,kernel_size,stride):
        super(Coarse_CU_Encoder, self).__init__()
        self.netu_x = Coarse_CSC_layer(num_iter=num_iter[0], in_channels=in_channel[0], num_filters=num_filters[0], kernel_size=kernel_size[0], stride=stride[0])
        self.netu_y = Coarse_CSC_layer(num_iter=num_iter[1], in_channels=in_channel[1], num_filters=num_filters[1], kernel_size=kernel_size[1], stride=stride[1])
        self.netc = Coarse_CSC_layer(num_iter=num_iter[2], in_channels=in_channel[2], num_filters=num_filters[2], kernel_size=kernel_size[2], stride=stride[2])
    def forward(self,X,Y):
        A1_u, X_u  = self.netu_x(X)
        B1_u, Y_u = self.netu_y(Y)
        X_c = X - X_u
        Y_c = Y - Y_u
        XY_c = torch.cat((X_c, Y_c), dim=1)
        AB1_c,_ = self.netc(XY_c)
        return AB1_c, A1_u, B1_u

class Fine_CU_Encoder(nn.Module):
    def __init__(self,num,num_iter,in_channel,num_filters,kernel_size,stride):
        super(Fine_CU_Encoder, self).__init__()
        self.netu_x = Fine_CSC_layer(num_iter=num_iter[num], in_channels=in_channel[num], num_filters=num_filters[num], kernel_size=kernel_size[num], stride=stride[num])
        self.netu_y = Fine_CSC_layer(num_iter=num_iter[num], in_channels=in_channel[num], num_filters=num_filters[num], kernel_size=kernel_size[num], stride=stride[num])
        self.netc = Fine_CSC_layer(num_iter=num_iter[num], in_channels=3*in_channel[num], num_filters=num_filters[num], kernel_size=kernel_size[num], stride=stride[num])
    def forward(self,C_in,U_in,V_in):
        U_out, U_re = self.netu_x(U_in)
        V_out, V_re = self.netu_y(V_in)
        U_c = U_in - U_re
        V_c = V_in - V_re
        UVC = torch.cat((U_c, V_c, C_in), dim=1)
        C_out,_ = self.netc(UVC)
        return C_out, U_out, V_out

class Map_final(nn.Module):
    def __init__(self,in_channel,num_filters,kernel_size):
        super(Map_final, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.num_filters = num_filters
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters, kernel_size=self.kernel_size, padding=self.padding, stride=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=self.kernel_size, padding=self.padding, stride=2, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight.data)
        nn.init.xavier_uniform_(self.conv2.weight.data)
    def forward(self, C):
        C = F.relu(self.conv2(F.relu(self.conv1(C))))
        return C

class CU_Encoder(nn.Module):
    def __init__(self,num_layer,num_iter,in_channel,num_filters,kernel_size,stride):
        super(CU_Encoder, self).__init__()
        self.num_layer = num_layer
        self.coarse_encoder = Coarse_CU_Encoder(num_iter,in_channel,num_filters,kernel_size,stride)
        self.fine_encoder: nn.ModuleList = nn.ModuleList()
        # self.final_map = Map_final(num_filters[-1], num_filters[-1]*2, 3)
        for i in range(self.num_layer):
            num = 3+i
            self.fine_encoder.append(Fine_CU_Encoder(num,num_iter,in_channel,num_filters,kernel_size,stride))
    def forward(self,X,Y):
        C1, U1, V1 = self.coarse_encoder(X, Y)
        C, U, V = C1, U1, V1
        for i in range(self.num_layer):
            C, U, V = self.fine_encoder[i](C, U, V)
        return C

class MMCSC(nn.Module):
    def __init__(self,num_layer, num_class,channel_per_class,down_scale_encoder,down_scale_classifier,x_in_channels,y_in_channels,c_out_channels,x_out_channels,y_out_channels, need_gamma,k_c,k_f,J,M):
        super(MMCSC, self).__init__()
        self.need_gamma = need_gamma
        self.num_layer = num_layer
        self.num_class = num_class
        self.channel_per_class = channel_per_class
        self.down_scale_encoder = down_scale_encoder
        self.down_scale_classifier = down_scale_classifier #image_size = down_scale_encoder*down_scale_classifier
        self.x_in_channels = x_in_channels  # input RGB image channel numbers
        self.y_in_channels = y_in_channels # input Depth image channel numbers
        self.x_out_channels = x_out_channels # filter channel numbers during ista for RGB inputs
        self.y_out_channels = y_out_channels # filter channel numbers during ista for Depth inputs
        self.c_out_channels = c_out_channels # filter channel numbers during ista for Common inputs

        self.k_c = k_c
        self.k_f = k_f
        self.J = J
        self.M = M
        self.num_iter = [self.J, self.J, self.J]
        self.in_channel = [self.x_in_channels, self.y_in_channels, self.x_in_channels + self.y_in_channels, self.c_out_channels]
        self.num_filters = [self.x_out_channels, self.y_out_channels, self.c_out_channels]
        self.kernel_size = [self.k_c, self.k_c, self.k_c]
        self.stride = [1,1,1]
        times = int(math.pow(self.down_scale_encoder,1/self.num_layer))
        for i in range(self.num_layer):
            self.num_iter.append(self.J)
            self.kernel_size.append(self.k_f)
            self.stride.append(times)
            if i == self.num_layer-1:
                self.in_channel.append(self.num_class * (times ** (i + 1)) * self.M// self.down_scale_encoder)#
                self.num_filters.append(self.num_class * (times ** (i + 1)) * self.M// self.down_scale_encoder)  #
            else:
                self.in_channel.append(self.num_class * (times ** (i + 1)) // (self.down_scale_encoder // 4))  #
                self.num_filters.append(self.num_class * (times ** (i + 1)) // (self.down_scale_encoder // 4))  #
        self.padding = (self.kernel_size[-1] - 1) // 2
        self.encoder = CU_Encoder(num_layer = self.num_layer,num_iter=self.num_iter,in_channel=self.in_channel,num_filters=self.num_filters,kernel_size=self.kernel_size,stride=self.stride)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.num_class*self.M, self.num_class)

    def forward(self,X):
        c = X.shape[1]
        X, Y = X[:, :3, :, :], X[:, 3:c, :, :]
        Gamma2 = self.encoder(X,Y)
        # print(Gamma2.shape)
        Gamma =self.avg_pool(Gamma2)
        # print(Gamma.shape)
        Gamma = Gamma.view(Gamma.shape[0], -1)
        # print(Gamma.shape)
        out_class = self.classifier(Gamma)
        out_class = F.log_softmax(out_class, dim=1)
        if self.need_gamma:
            return out_class, [Gamma2]
        else:
            return out_class
#为了不损失数据，最好取channel_per_class = down_scale_encoder 即图片大小缩小多少倍，对应channel数增加多少倍
#num_layer和down_scale_encoder共同决定fineCSC层的stride，即如果3层，缩小8倍，则每一层缩小2倍，因此down_scale_encoder最好是num_layer的整数次方
#根据输入224，down_scale_encoder最大为32

# Number of parameters:           2.22 M
#n1,d8,cx32,cy32
#num_layer=1,channel_per_class=down_scale_encoder=8,x_out_channels=32,y_out_channels=32
def cu_mmcsc_n1_d8(num_layer=1,num_class=51,channel_per_class=8,down_scale_encoder=8,down_scale_classifier=16,x_in_channels=3,y_in_channels=3,c_out_channels=32,x_out_channels=32,y_out_channels=32, need_gamma=False,k_c=7,k_f=4,J=4,M=4):
    model = MMCSC(num_layer=num_layer,num_class=num_class,channel_per_class=channel_per_class,down_scale_encoder=down_scale_encoder,down_scale_classifier=down_scale_classifier,x_in_channels=x_in_channels,y_in_channels=y_in_channels,c_out_channels=c_out_channels,x_out_channels=x_out_channels,y_out_channels=y_out_channels, need_gamma=need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
    return model

# Number of parameters:           4.36 M
#n1,d16,cx32,cy32
def cu_mmcsc_n1_d16(num_layer=1,num_class=51,channel_per_class=16,down_scale_encoder=16,down_scale_classifier=16,x_in_channels=3,y_in_channels=3,c_out_channels=32,x_out_channels=32,y_out_channels=32, need_gamma=False,k_c=7,k_f=4,J=4,M=4):
    model = MMCSC(num_layer=num_layer,num_class=num_class,channel_per_class=channel_per_class,down_scale_encoder=down_scale_encoder,down_scale_classifier=down_scale_classifier,x_in_channels=x_in_channels,y_in_channels=y_in_channels,c_out_channels=c_out_channels,x_out_channels=x_out_channels,y_out_channels=y_out_channels, need_gamma=need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
    return model

# Computational complexity:       144.54 GMac
# Number of parameters:           8.64 M
#n1,d32,cx32,cy32
def cu_mmcsc_n1_d32(num_layer=1,num_class=51,channel_per_class=32,down_scale_encoder=32,down_scale_classifier=16,x_in_channels=3,y_in_channels=3,c_out_channels=32,x_out_channels=32,y_out_channels=32, need_gamma=False,k_c=7,k_f=4,J=4,M=4):
    model = MMCSC(num_layer=num_layer,num_class=num_class,channel_per_class=channel_per_class,down_scale_encoder=down_scale_encoder,down_scale_classifier=down_scale_classifier,x_in_channels=x_in_channels,y_in_channels=y_in_channels,c_out_channels=c_out_channels,x_out_channels=x_out_channels,y_out_channels=y_out_channels, need_gamma=need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
    return model

# Computational complexity:       43.79 GMac
# Number of parameters:           9.07 M
#n3,d8,cx32,cy32
def cu_mmcsc_n3_d8(num_layer=3,num_class=51,channel_per_class=8,down_scale_encoder=8,down_scale_classifier=16,x_in_channels=3,y_in_channels=3,c_out_channels=32,x_out_channels=32,y_out_channels=32, need_gamma=False,k_c=7,k_f=4,J=4,M=4):
    model = MMCSC(num_layer=num_layer,num_class=num_class,channel_per_class=channel_per_class,down_scale_encoder=down_scale_encoder,down_scale_classifier=down_scale_classifier,x_in_channels=x_in_channels,y_in_channels=y_in_channels,c_out_channels=c_out_channels,x_out_channels=x_out_channels,y_out_channels=y_out_channels, need_gamma=need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
    return model

# Computational complexity:       47.35 GMac
# Number of parameters:           14.67 M
#n2,d16,cx32,cy32
def cu_mmcsc_n2_d16(num_layer=2,num_class=51,channel_per_class=16,down_scale_encoder=16,down_scale_classifier=16,x_in_channels=3,y_in_channels=3,c_out_channels=32,x_out_channels=32,y_out_channels=32, need_gamma=False,k_c=7,k_f=4,J=4,M=4):
    model = MMCSC(num_layer=num_layer,num_class=num_class,channel_per_class=channel_per_class,down_scale_encoder=down_scale_encoder,down_scale_classifier=down_scale_classifier,x_in_channels=x_in_channels,y_in_channels=y_in_channels,c_out_channels=c_out_channels,x_out_channels=x_out_channels,y_out_channels=y_out_channels, need_gamma=need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
    return model

# Computational complexity:       55.46 GMac
# Number of parameters:           36.06 M
#n4,d16,cx32,cy32
def cu_mmcsc_n4_d16(num_layer=4,num_class=51,channel_per_class=16,down_scale_encoder=16,down_scale_classifier=16,x_in_channels=3,y_in_channels=3,c_out_channels=32,x_out_channels=32,y_out_channels=32, need_gamma=False,k_c=7,k_f=4,J=4,M=4):
    model = MMCSC(num_layer=num_layer,num_class=num_class,channel_per_class=channel_per_class,down_scale_encoder=down_scale_encoder,down_scale_classifier=down_scale_classifier,x_in_channels=x_in_channels,y_in_channels=y_in_channels,c_out_channels=c_out_channels,x_out_channels=x_out_channels,y_out_channels=y_out_channels, need_gamma=need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
    return model

# Computational complexity:       66.5 GMac
# Number of parameters:           143.98 M
#n5,d32,cx32,cy32
def cu_mmcsc_n5_d32(num_layer=5,num_class=51,channel_per_class=32,down_scale_encoder=32,down_scale_classifier=16,x_in_channels=3,y_in_channels=3,c_out_channels=32,x_out_channels=32,y_out_channels=32, need_gamma=False,k_c=7,k_f=4,J=4,M=4):
    model = MMCSC(num_layer=num_layer,num_class=num_class,channel_per_class=channel_per_class,down_scale_encoder=down_scale_encoder,down_scale_classifier=down_scale_classifier,x_in_channels=x_in_channels,y_in_channels=y_in_channels,c_out_channels=c_out_channels,x_out_channels=x_out_channels,y_out_channels=y_out_channels, need_gamma=need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
    return model

if __name__ == '__main__':
    net = cu_mmcsc_n3_d8()
    print(net)
    print('# network parameters:', sum(param.numel() for param in net.parameters()) / 1e6, 'M')
    a = torch.rand([10, 6, 224, 224])
    b = torch.rand([10, 6, 224, 224])
    a_out=net(a)
    print(a_out.shape)
    # lis=torch.max(a_out, dim=1)[1]
    # print(torch.eq(torch.max(a_out, dim=1)[1], val_labels.to(device)).sum().item())
    macs, params = get_model_complexity_info(net, (6, 224, 224), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))