#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import PreEmphasis
import time

print('#############################')
print('    DTDY_ResNet34_quarter    ')
print('#############################')

def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNet(BasicBlock_DTDY, [2, 2, 2, 2], num_filters, nOut, **kwargs)
    return model 


class ResNet(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='ASP', n_mels=40, log_input=True, **kwargs):
        super(ResNet, self).__init__() 

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input
        self.outmap_size = n_mels

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.outmap_size = int(self.outmap_size/2)
        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=1)

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb        = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)
                )

        outmap_size = int(self.n_mels/8)
        
        if self.encoder_type == "SAP":
            self.attention = nn.Sequential(
                nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
                nn.Softmax(dim=2),
                )
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            self.attention = nn.Sequential(
                nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
                nn.Softmax(dim=2),
                )
            out_dim = num_filters[3] * outmap_size * 2
        elif self.encoder_type == "AVG":
            self.sap_linear = nn.AdaptiveAvgPool1d(1)
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "STP":
            self.sap_linear = nn.AdaptiveAvgPool1d(1)
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                DTDY_conv2d(self.inplanes, planes * block.expansion, self.outmap_size, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.outmap_size, stride, downsample))
        self.inplanes = planes * block.expansion
        if stride != 1:
            self.outmap_size = int(self.outmap_size/2)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.outmap_size))

        return nn.Sequential(*layers)


    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.kaiming_normal_(out)
        return out

    def forward(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1)
                
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0],-1,x.size()[-1])

        if self.encoder_type == "SAP":
            w = self.attention(x)
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            w = self.attention(x)
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,sg),1)
        elif self.encoder_type == "AVG":
            x = self.sap_linear(x)
        elif self.encoder_type == "STP":
            mu = torch.mean(x, dim=2)
            sg = torch.sqrt( ( torch.mean((x**2), dim=2) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,sg),1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

################ Module ####################

class BasicBlock_DTDY(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, map_size, stride=1, downsample=None):
        super(BasicBlock_DTDY, self).__init__()
        self.conv1 = DTDY_conv2d(inplanes, planes, map_size = map_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1:
            map_size = int(map_size/2)

        self.conv2 = DTDY_conv2d(planes, planes, map_size = map_size, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
        

class DTDY_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, map_size, kernel_size, stride=1, padding=0, bias=False):
        super(DTDY_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.map_size = map_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
    

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.dim = int(math.sqrt(in_planes*2 + out_planes*2))

        self.q = nn.Conv2d(in_planes, self.dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.p = nn.Conv2d(self.dim, out_planes, kernel_size = 1, stride = 1, padding = 0, bias=False)

        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)

        self.fc_phi = fc_phi(in_planes, map_size, self.dim , kernel_size, stride, padding)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        
        r = self.conv(x)

        phi = self.fc_phi(x)

        out = self.bn1(self.q(x))
        out = self.bn2(torch.einsum('bdkw, bkhw -> bdhw', phi, out)) + out
        out = self.p(out)
        out = out + r
         
        return out


class fc_phi(nn.Module):
    def __init__(self, in_planes, map_size, dim, kernel_size, stride, padding):
        super(fc_phi, self).__init__()

        self.dim = dim
                
        start_planes = in_planes + map_size
        hidden_planes = (self.dim ** 2) // 8 

        self.fc1 = nn.Conv1d(start_planes, hidden_planes, kernel_size, stride = stride, padding=padding)
        self.semodule = SEModule_small(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, dim**2, 3, padding = 1) 
        self.nonlinear = Hsigmoid(inplace=True)

        self.norm = nn.LayerNorm(dim**2, elementwise_affine = False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        b, _, _, _= x.size()

        x_c = torch.mean(x, dim=2)
        x_f = torch.mean(x, dim=1)
        x = torch.cat([x_c,x_f], dim=1)
        
        x = self.fc1(x)
        x = self.semodule(x)
        x = self.fc2(x)
        x = self.nonlinear(x)
        
        x = x.view(b, self.dim, self.dim, -1)

        return x


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size = 1, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

