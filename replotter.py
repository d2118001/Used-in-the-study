#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import csv
import numpy as np
import os
import sys
import glob
import tqdm

#元の行数
SOURCE_LENGTH = 0
#線形補完後の行数
OUTPUT_LENGTH = 0

row=0      #gyou
column=0   #retsu

def LoadData(data_path):
    #global row
    #global column
    data_training_tmp = np.loadtxt(data_path,delimiter = ',')
    #column = len(data_training_tmp[0])
    #row = sum(1 for line in open(data_path))
    data_tmp =[line for line in data_training_tmp]
    """
    data_tmp = []
    for line in data_training_tmp:
        # for data in line:
        data_tmp.append(line)
    """
    return np.array(data_tmp,dtype=np.float32)

def listReplot(sourceData):
    global SOURCE_LENGTH
    global OUTPUT_LENGTH
    elements = len(sourceData[0])
    SOURCE_LENGTH = len(sourceData)
    OUTPUT_LENGTH = 20
    #線形補間されたデータを格納する配列
    plotteddata = np.empty((OUTPUT_LENGTH, elements),dtype=np.float32)
    #線形補間してplotteddataに入れる
    plotteddata = Replot(sourceData,plotteddata)
    return plotteddata

def signReplot(sourceData,magnifaction):
    global SOURCE_LENGTH
    global OUTPUT_LENGTH
    elements = len(sourceData[0])
    SOURCE_LENGTH = len(sourceData)
    OUTPUT_LENGTH = int(SOURCE_LENGTH * magnifaction)
    #線形補間されたデータを格納する配列
    plotteddata = np.empty((OUTPUT_LENGTH, elements),dtype=np.float32)
    #線形補間してplotteddataに入れる
    plotteddata = Replot(sourceData,plotteddata)
    return plotteddata

def ReplotInitOnce(filepath):
    global SOURCE_LENGTH
    global OUTPUT_LENGTH
    SOURCE_LENGTH = 100
    OUTPUT_LENGTH = 20
    #要素数(列数)
    elements = 72
    #線形補間されたデータを格納する配列(0で初期化)
    #線形補間してplotteddataに入れる
    plotteddata = np.empty((OUTPUT_LENGTH, elements),dtype=np.float32)
    receiveddata = LoadData(filepath)
    plotteddata = Replot(receiveddata,plotteddata)
    return plotteddata

def ReplotInit(filepath):
    global SOURCE_LENGTH
    global OUTPUT_LENGTH
    SOURCE_LENGTH = 100
    OUTPUT_LENGTH = 20
    #要素数(列数)
    elements = 72
    #線形補間されたデータを格納する配列(0で初期化)
    #線形補間してplotteddataに入れる
    for signPath in tqdm.tqdm(glob.glob(filepath+"*")):
        plotteddata = np.empty((OUTPUT_LENGTH, elements))
        receiveddata = LoadData(signPath)
        plotteddata = Replot(receiveddata,plotteddata)
        FileWriter(plotteddata,filepath+os.path.basename(signPath))
    return

def Replot(ps, po):
    #出力する座標の原点からの距離(全体の距離はSOURCE_LENGTH * OUTPUT_LENGTH)
    for i in range(OUTPUT_LENGTH):
        distance = (SOURCE_LENGTH - 1) * i

        #このdistanceまでに元の座標が何個あるか
        lastSource = 0

        #元の座標の最後のものからの距離(全体の距離はSOURCE_LENGTH * OUTPUT_LENGTH)
        distanceFromLastSource = distance

        while distanceFromLastSource > OUTPUT_LENGTH - 1:
            distanceFromLastSource -= OUTPUT_LENGTH - 1
            lastSource += 1

        if lastSource < SOURCE_LENGTH - 1:
            for j in range(len(ps[0])):
                po[i, j] = ps[lastSource, j] + (ps[lastSource + 1, j] - ps[lastSource, j]) / (OUTPUT_LENGTH - 1) * distanceFromLastSource

        else :
            for j in range(len(ps[0])):
                po[i, j] = ps[lastSource, j]
    return po

def FileWriter(result,filepath):
    #result = np.array(result,dtype=np.float32)
    np.savetxt(filepath,result,fmt="%.6f",delimiter=",")

if __name__ == '__main__':
    ReplotInit()