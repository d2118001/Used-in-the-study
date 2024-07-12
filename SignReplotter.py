from __future__ import print_function
import argparse
import time
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda as xp
import csv
import numpy as np
import tqdm
import sys
import os
import random
from chainer import optimizers
from chainer import training,Variable
from chainer.training import extensions
from sklearn.metrics import classification_report
import cnn_1ch_nw_def as cnn_def
import glob
import shutil
import gc
import replotter
row = 0 #行
column = 0  #列

def loadData(data_path):
    global row ,column
    csvfile = np.loadtxt(data_path,delimiter = ',',skiprows=1)
    column = len(csvfile[0])
    row = len(csvfile)
    tmp = [line for line in csvfile]
    return np.array(tmp,dtype=np.float32)

def shaping(raw_data):
    row = 20
    column = 72
    d = []
    for i in range(len(raw_data)):
        tmp1 = []
        for j in range(row):
            tmp2 = [[]]
            for k in range(column):
                tmp2[0].append(raw_data[i][j][k])
            tmp1.append(tmp2)
        d.append(tmp1)
    data = np.asarray(d,dtype=np.float32)
    data = np.reshape(data,(len(data),-1,row,column))
    return data

def main():
    sign = loadData("./kiuchi/world.csv")
    #sign = loadData("./kiuchi/zero.csv")
    for i in range(16):
        magnification = i * 0.2
        replotedSign = Replotter.signReplot(sign,magnification)
        for shiftPoint in range(100):
            pointCount20 = 0
            pointCount100 = 0
            dividedSign 20
            if pointCount20 == 19 and shiftPoint < 20:

            else:

                pointCount20+=1
            if pointCount20 == 19 and shiftPoint < 20:


if __name__ == '__main__':
    main()