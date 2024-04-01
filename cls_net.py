import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import math
import os
import glob
import torch.nn.functional as F
import torch
import cv2
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import math
import os
import torchvision.models as models
import basic_net as basic_net
# from ResCBAMIQA import *
import torch.cuda
from ptflops import get_model_complexity_info

"---------------- CLS Net ( Backbones+FCN ) --------------------------------------"
class clsNet(nn.Module):

    def __init__(self, classes_num=5, basic_model='resnet18',pretrain=True,need_gamma=False,k_c=7,k_f=4,J=4,M=4):
        super(clsNet, self).__init__()
        self.basic_model=basic_model
        self.need_gamma = need_gamma
        if self.basic_model == 'resnet18':
            self.resNet1 = basic_net.resnet18(pretrained=pretrain)
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, classes_num)
        elif self.basic_model == 'resnet101':
            self.resNet1 = basic_net.resnet101(pretrained=pretrain)
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, classes_num)
        elif self.basic_model == 'resnet101_rgbd':
            self.resNet1 = basic_net.resnet101_rgbd(pretrained=pretrain)
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, classes_num)
        elif self.basic_model == 'inception':
            self.inception1 = basic_net.inception_v3(pretrained=pretrain)
            self.inception = list(self.inception1.children())
            self.fc = nn.Linear(2048, classes_num)
        elif self.basic_model == 'densenet':
            self.densenet1 = basic_net.densenet121(pretrained=pretrain)
            self.densenet = list(self.densenet1.children())
            self.fc = nn.Linear(1024, classes_num)
        elif self.basic_model == 'resnet_cbam':
            self.resNet1 = resnet34_cbam(pretrained=pretrain)
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(10752, classes_num)
        elif self.basic_model == 'mlista':
            self.MLISTA = basic_net.ml_ista(T=6,classes_num = classes_num)
        elif self.basic_model == 'mlista_rgbd':
            self.MLISTA = basic_net.ml_ista_rgbd(T=6,classes_num = classes_num)
        elif self.basic_model == 'mlfista_rgbd':
            self.MLISTA = basic_net.ml_fista_rgbd(T=6, classes_num=classes_num)
        elif self.basic_model == 'mllista_rgbd':
            self.MLISTA = basic_net.ml_lista_rgbd(T=6, classes_num=classes_num)
        elif self.basic_model == 'lbpnet':
            self.LBP = basic_net.lbp_net(T=6,classes_num = classes_num)
        elif self.basic_model == 'cnn_transnet':
            self.CNN_TransNet = basic_net.cnn_transnet(pretrained=pretrain)
        elif 'mmcsc' in self.basic_model:
            if self.basic_model == 'mmcsc_n1_d8_cx32_cy32':
                self.MMCSC = basic_net.mmcsc_n1_d8_cx32_cy32(num_class = classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'mmcsc_n1_d16_cx32_cy32':
                self.MMCSC = basic_net.mmcsc_n1_d16_cx32_cy32(num_class=classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'mmcsc_n1_d32_cx32_cy32':
                self.MMCSC = basic_net.mmcsc_n1_d32_cx32_cy32(num_class = classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'mmcsc_n3_d8_cx32_cy32':
                self.MMCSC = basic_net.mmcsc_n3_d8_cx32_cy32(num_class = classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'mmcsc_n2_d16_cx32_cy32':
                self.MMCSC = basic_net.mmcsc_n2_d16_cx32_cy32(num_class = classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'mmcsc_n4_d16_cx32_cy32':
                self.MMCSC = basic_net.mmcsc_n4_d16_cx32_cy32(num_class=classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'mmcsc_n5_d32_cx32_cy32':
                self.MMCSC = basic_net.mmcsc_n5_d32_cx32_cy32(num_class = classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'cu_mmcsc_n1_d8_cx32_cy32':
                self.MMCSC = basic_net.cu_mmcsc_n1_d8_cx32_cy32(num_class = classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'cu_mmcsc_n3_d8_cx32_cy32':
                self.MMCSC = basic_net.cu_mmcsc_n3_d8_cx32_cy32(num_class = classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'cu_mmcsc_n1_d8':
                self.MMCSC = basic_net.cu_mmcsc_n1_d8(num_class = classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'cu_mmcsc_n3_d8':
                self.MMCSC = basic_net.cu_mmcsc_n3_d8(num_class = classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)
            elif self.basic_model == 'cu_mmcsc_n3_d8_share4':
                self.MMCSC = basic_net.cu_mmcsc_n3_d8_share4(num_class=classes_num, need_gamma=self.need_gamma,k_c=k_c,k_f=k_f,J=J,M=M)

    def embed(self, x):
        x = self.features(x)
        x = self.spatial_pyramid_pool(x, x.size(0), [int(x.size(2)),int(x.size(3))], [4,2,1])
        x = F.normalize(x)
        return x.squeeze()

    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):

        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if (i == 0):
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp

    def forward(self, x):
        if 'resnet' in self.basic_model:
            x = self.features(x)  # 512
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.basic_model == 'inception':
            x = self.inception[0](x)
            x = self.inception[1](x)
            x = self.inception[2](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.inception[3](x)
            x = self.inception[4](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.inception[5](x)
            x = self.inception[6](x)
            x = self.inception[7](x)
            x = self.inception[8](x)
            x = self.inception[9](x)
            x = self.inception[10](x)
            x = self.inception[11](x)
            x = self.inception[12](x)
            x = self.inception[14](x)
            x = self.inception[15](x)
            x = self.inception[16](x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = F.dropout(x, training=self.training)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.basic_model == 'densenet':
            features = self.densenet[0](x)  # 512
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
            out = self.fc(out)
            return out
        elif self.basic_model == 'resnet_cbam':
            x = self.features(x)
            x = self.spatial_pyramid_pool(x, x.size(0), [int(x.size(2)),int(x.size(3))], [4,2,1])
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif 'ista' in self.basic_model:
            x = self.MLISTA(x)
            return x
        elif self.basic_model == 'lbpnet':
            x = self.LBP(x)
            return x
        elif 'mmcsc' in self.basic_model:
            x = self.MMCSC(x)
            return x
        elif self.basic_model == 'cnn_transnet':
            out,x = self.CNN_TransNet(x)
            return out


if __name__ == "__main__":

    x = torch.randn((10, 6, 224, 224))

    net = clsNet(classes_num=51, basic_model="mmcsc_n4_d16_cx32_cy32", pretrain=False)
    # net = clsNet(classes_num=51, basic_model="mlista", pretrain=False)
    out = net(x)

    macs, params = get_model_complexity_info(net, (6, 224, 224), as_strings=True,print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('# network parameters:', sum(param.numel() for param in net.parameters()) / 1e6, 'M')
    print(out.shape)