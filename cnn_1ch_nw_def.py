#!/usr/bin/env python
# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
k = 2
p = 0
chainer.global_config.autotune = True
#chainer.global_config.cudnn_fast_batch_normalization = True
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):

        super(MLP, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 256, ksize=k, pad=p)
            self.conv2 = L.Convolution2D(None, 128, ksize=k, pad=p)
            self.conv3 = L.Convolution2D(None, 64, ksize=k, pad=p)
            #self.fc1 = L.Linear(None, 128)
            self.fc2 = L.Linear(None, n_out)
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(64)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pooling_2d(x, ksize=k)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pooling_2d(x, ksize=k)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pooling_2d(x, ksize=k)
        #x = self.fc1(x)
        #x = F.dropout(x)
        x = self.fc2(x)
        return x