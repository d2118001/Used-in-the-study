#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from unittest import result
import cupy as cp
import argparse
import time
import csv
import numpy as np
import os
import sys
import random
import glob
import pickle

from friday13 import readLogJson
from friday13 import appendAveStdtoAlllog
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import confusion_matrix
from sklearn import svm
import Pytorch_cnn_1ch_nw_def as nw_def
#import igniter as ig
from matplotlib import pyplot as plt
import pandas as pd
#from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

row=0     #gyou
column=0  #retsu

iterationResult = []
allSubjecttpfpfntn = []
allSubjectScore = []
all_pickle_data = []
all_pickle_label = []

def load_data(dataPath):
    global row
    global column
    loadedText = np.loadtxt(dataPath,delimiter = ',')
    column = len(loadedText[0])
    row = sum(1 for line in open(dataPath))
    return np.array(loadedText,dtype=np.float32)

def openPickel(file_path):
    with open(file_path, mode='rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

def shaping(rawData):
    data = np.asarray(rawData,dtype=np.float32)
    data = np.reshape(data,(len(data),-1,row,column))
    cp.asarray(data)
    return data

def calculateScore(cm,subjectCount):
    global allSubjectScore
    global allSubjecttpfpfntn
    tp = []
    tn = []
    fp = []
    fn = []
    precision = []
    recall = []
    fscore = []
    far = []
    frr = []
    eer = []
    tmp = np.sum(cm,axis=0)
    for i in range(subjectCount):
        tp.append(cm[i][i])
        fp.append(np.sum(cm[i])-tp[i])
        fn.append(tmp[i]-tp[i])
    for i in range(subjectCount):
        tn.append(np.sum(tp)-tp[i])
    for i in range(subjectCount):
        pre = tp[i]/(tp[i]+fp[i])
        rec = tp[i]/(tp[i]+fn[i])
        fs = (2*pre*rec)/(pre+rec)
        fr = fn[i]/(fn[i]+tp[i])
        fa = fp[i]/(tn[i]+fp[i])
        fscore.append(fs)
        precision.append(pre)
        recall.append(rec)
        frr.append(fr)
        far.append(fa)
        eer.append((fr+fa)/2)
    for i in range(subjectCount):
        allSubjecttpfpfntn[i].append(np.array([tp[i],tn[i],fp[i],fn[i]],dtype=np.int32))
        allSubjectScore[i].append(np.array([precision[i],recall[i],fscore[i],far[i],frr[i],eer[i]],dtype=np.float32))
    return

def main(data_dir,testCount,result_path):
    parser = argparse.ArgumentParser(description='Pytorch sign verification system by Kamaishi')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()
    """
    print('Leap motion Verification System by "Kamaishi"')
    print(f'GPU: {args.gpu}')
    print(f'# unit: {args.unit}')
    print(f'# Minibatch-size: {args.batchsize}')
    print(f'# epoch: {args.epoch}')
    print('')
    """

    unitOut = len(glob.glob(data_dir+"/*"))
    unitOut=4
    # -----データここから-----
    start = time.time()
    # テストデータの割合の分母
    trainDataRate = 3
    
    #Pickleを使う読み込み（速い）
    data_train_tmp = []
    label_train_tmp = []
    data_test_tmp = []
    label_test_tmp = []
    data_valid_tmp = []
    label_valid_tmp = []

    #出力ユニット数と同じ数(方向の数)繰り返す
    for i in range(unitOut):
        tmp = []
        #とりあえず1方向のサンプルを順番で格納していく(後でシャッフルする)
        for j in range(len(all_pickle_label[i])):
            tmp.append(all_pickle_data[i][j])
        random.shuffle(tmp) #サンプルの順番をシャッフルしてサンプルが偏らないようにする

        #trainDataRateでtmpを分割(t1,t2が学習用)
        t1,t2,t3 = np.array_split(np.array(tmp),trainDataRate)
        #t1 = np.append(t1,t2,axis = 0) #t1,t2をくっつける
        data_test_tmp.extend(t3)
        data_valid_tmp.extend(t2)
        data_train_tmp.extend(t1)

        #ラベル貼り
        for j in range(len(t1)):
            label_train_tmp.append(str(i))
            label_valid_tmp.append(str(i))
        for j in range(len(t3)):
            label_test_tmp.append(str(i))

    elapsed_time = time.time() - start
    print (f"\nfile read time:{elapsed_time}[sec]")
    start = time.time()

    # データを4次元にしないと機械学習できないけどpickleでやってある(サンプル数,チャネル数,行,列)
    # 訓練,テストデータ(ラベルも)をPytorch用に変換
    train_label = np.array(label_train_tmp,dtype=np.int32)
    train_data = np.array(data_train_tmp,dtype=np.float32)
    train_data = np.squeeze(train_data)
    train_data = np.reshape(train_data,(len(train_label),1440))

    test_label = np.array(label_test_tmp,dtype=np.int32)
    test_data = np.array(data_test_tmp,dtype=np.float32)
    test_data = np.squeeze(test_data)    
    test_data = np.reshape(test_data,(len(test_label),1440))

    valid_label = np.array(label_valid_tmp,dtype=np.int32)
    valid_data = np.array(data_valid_tmp,dtype=np.float32)
    valid_data = np.squeeze(valid_data)    
    valid_data = np.reshape(valid_data,(len(valid_label),1440))

    #clf = RandomForestClassifier(n_jobs = 4)
    #clf = svm.SVC()
    clf = GradientBoostingClassifier()

    clf.fit(train_data, train_label)
    prediction_label = clf.predict(test_data)

    #print(prediction_label)
    print(clf.score(test_data,test_label))

    cm = confusion_matrix(test_label,prediction_label,labels=[i for i in range(unitOut)])
    #PlotConfusionMatrix(cm,labels=[i for i in range(unitOut)])
    report = classification_report(test_label, prediction_label, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(f"{result_path}/{int(testCount)}_report.csv")
    elapsed_time = time.time() - start
    print (f"training_time:{elapsed_time}[sec]")
    return

def PlotConfusionMatrix(cm, labels):
    #import seaborn as sns
    #sns.set()

    df = pd.DataFrame(cm)
    df.index = labels
    df.columns = labels

    f, ax = plt.subplots()
    #f, ax = plt.figure()
    #sns.heatmap(df, annot=True, fmt="d", linewidths=.5, ax=ax)
    ax.set_ylim(len(labels), 0)
    #print(cm)
    #plt.show()

def oneFileWriter(result):
    result = np.array(result,dtype=np.float32)
    np.savetxt("./result/per_iteration/test.csv",result,fmt="%.6f",delimiter=",")
    with open('./result/per_iteration/test.csv','a')as f:
        writer = csv.writer(f)
        writer.writerow(["Precision","Recall","F-score","Acuraccy"])
        writer.writerow(["Average"])
        writer.writerow(np.mean(result, axis = 0))
        writer.writerow(["Standard deviation"])
        writer.writerow(np.std(result, axis = 0))
    return

def fileWriter(result1,result2,subjectNumber):
    filePath = "./result/per_subject/"+subjectNumber+"_"
    #result = np.array(result1,dtype=np.float32)
    #result = np.array(result2,dtype=np.float32)
    np.savetxt(filePath+"TP_TN_FP_FN.csv",result1,fmt="%.6f",delimiter=",")
    np.savetxt(filePath+"PRF_FAR_FRR_EER.csv",result2,fmt="%.6f",delimiter=",")
    with open(filePath+"TP_TN_FP_FN.csv",'a')as f:
        writer = csv.writer(f)
        writer.writerow(["TP","TN","FP","FN"])
        writer.writerow(["Average"])
        writer.writerow(np.mean(result1, axis = 0))
        writer.writerow(["Standard deviation"])
        writer.writerow(np.std(result1, axis = 0))
    with open(filePath+"PRF_FAR_FRR_EER.csv",'a')as f:
        writer = csv.writer(f)
        writer.writerow(["Precision","Recall","F-score","FAR","FRR","EER"])
        writer.writerow(["Average"])
        writer.writerow(np.mean(result2, axis = 0))
        writer.writerow(["Standard deviation"])
        writer.writerow(np.std(result2, axis = 0))
    return

#方向の数に合わせて結果を格納するリストを作る
def setSubjectCount(subjectCount, pickle_path):
    global allSubjectScore
    global allSubjecttpfpfntn
    for i in range(subjectCount):
        allSubjectScore.append([])
        allSubjecttpfpfntn.append([])

    global all_pickle_label
    global all_pickle_data
    for i in range(directioncount):
        all_pickle_data.append(openPickel(f'{pickle_path}/Direction{i}_data.pickle'))
        all_pickle_label.append(openPickel(f'{pickle_path}/Direction{i}_label.pickle'))
    #all_pickle_dataは[方向][サンプル1...サンプルx][チャネル数(1)][行][列]
    return

if __name__ == '__main__':
    testcount = 10
    #os.makedirs("./result/per_iteration/",exist_ok=True)
    pickle_path = "directionpickle"
    result_path = "directionresultGBDT"
    os.makedirs(result_path, exist_ok=True)

    directioncount = 4
    setSubjectCount(directioncount, pickle_path)

    allstart = time.time()
    for i in range(testcount):
        main(pickle_path,i,result_path)
        #Test(oneTest,iter)
        #readLogJson(i)
    """
    for i in range(directioncount):
        fileWriter(allSubjecttpfpfntn[i],allSubjectScore[i],str(i))
    #oneFileWriter(iterationResult)
    """
    allelapsed_time = time.time() - allstart
    print(result_path)
    print (f"allelapsed_time:{allelapsed_time}[sec]")
    #appendAveStdtoAlllog()
    #os.system("dot_convert_to_png.bat")