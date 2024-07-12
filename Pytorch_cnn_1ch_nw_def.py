#!/usr/bin/env python
# -*- coding: utf-8 -*-
#import chainer
#import chainer.functions as F
#import chainer.links as L

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

k = 1

class MLP(nn.Module):
    def __init__(self, n_out):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, k,1,1)
        self.conv2 = nn.Conv2d(256, 128, k,1,1)
        #self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(233472, n_out)
        self.bn1   = nn.BatchNorm2d(256)
        self.bn2   = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, k)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, k)
        #x = self.fc1(x)
        #x = F.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x