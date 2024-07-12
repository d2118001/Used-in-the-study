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
import pickle
from chainer import optimizers
from chainer import training,Variable
from chainer.training import extensions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#import cnn_1ch_nw_def as cnn_def
import fc_1ch_nw_def as cnn_def
import glob
import cupy as cp
import concurrent.futures
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as prfs
import bincounter2 as bc

GPU = True
GPU_ID = 0
row=0      #gyou
column=0   #retsu
chainer.global_config.train = False

eval_flag=["recall","fscore"]

predictionLabel = []
userLabel = []
all_csv_data = []
all_csv_name = []
super_avg_max = [[],[],[],[],[],[],[],[],[],[],[]]
max_sa = [[],[],[],[],[],[],[],[],[],[],[]]
identification_mode = [0,1] #0:本人識別,1:未登録者識別
#identification_mode = [1] #0:本人識別,1:未登録者識別
imode = 0
ten_mode = [True,False] #10個だけ取り出すか
#ten_mode = [True]

def calculateScore(cm,subject_count,test_count):
    np.seterr(all='ignore')
    tp,tn,fp,fn,precision,recall,fscore,far,frr,eer = [],[],[],[],[],[],[],[],[],[]
    tmp = np.sum(cm,axis=0)
    for i in range(subject_count):
        tp.append(cm[i][i])
        fp.append(tmp[i]-tp[i])
        fn.append(np.sum(cm[i])-tp[i])
    for i in range(subject_count):
        tn.append(np.sum(tp)-tp[i])
    for i in range(subject_count):
        if i == test_count:
            pre,rec,fs,fr,fa = 0,0,0,0,0
        else:
            pre = tp[i]/(tp[i]+fp[i])
            if np.isnan(pre):
                pre = 0
            rec = tp[i]/(tp[i]+fn[i])
            if np.isnan(rec):
                rec = 0
            #fs = (2*pre*rec)/(pre+rec)
            fs = (2*1*rec)/(1+rec)
            if np.isnan(fs):
                fs = 0
            fr = fn[i]/(fn[i]+tp[i])
            fa = fp[i]/(tn[i]+fp[i])
        fscore.append(fs)
        precision.append(pre)
        recall.append(rec)
        frr.append(fr)
        far.append(fa)
        eer.append((fr+fa)/2)
    """
    for i in range(subject_count):
        if i == test_count:
            continue
    """
    return fscore, recall

def load_data(dataPath):
    global row
    global column
    loadedText = np.loadtxt(dataPath,delimiter = ',')
    column = len(loadedText[0])
    row = sum(1 for line in open(dataPath))
    return np.array(loadedText,dtype=np.float32)

def LoadDatatoByte(dataPath):
    loadedText = np.loadtxt(dataPath,delimiter = ',',dtype=bytes)
    return np.array(loadedText)

def LoadSubjectList(data_path,skipRow):
    return np.array(np.loadtxt(data_path,delimiter = ',',dtype=bytes,skiprows=skipRow))

def labeling(subjectNumber,result,model_count):
    global userLabel
    global predictionLabel
    global super_avg_max
    global max_sa
    result = np.array(result)
    result = F.softmax(result).array
    #result = F.sigmoid(result).array
    ul = []
    pl = []
    max_avg = 0.0
    sa_avg = 0.0
    #result.data1行(1つのサンプル)ごとでsoftmaxが最大値の被験者をplにいれる
    for signResult in result:
        ul.append(subjectNumber)
        #pl.append(np.argmax(signResult))
        #if subjectNumber == np.argmax(signResult) and np.max(signResult) >= 0.9:
        if subjectNumber == np.argmax(signResult):
            pl.append(np.argmax(signResult))
        else:
            pl.append(model_count)
        max_avg += np.max(signResult)
        """
        #softmaxが最大の人が本人で閾値を下回っていたらtpをfnにする
        if np.max(signResult) < 0.97 and subjectNumber == np.argmax(signResult):
            pl.append(model_count)
        else:
            pl.append(np.argmax(signResult))
        """
        maxresult = np.max(signResult)
        signResult[np.argmax(signResult)]= -10000
        #print(maxresult-np.max(signResult))
        sa = maxresult-np.max(signResult)
        sa_avg += sa
        """
        if sa > 0.86 and subjectNumber == np.argmin(signResult):
            pl.append(np.argmin(signResult))
        else:
            pl.append(model_count)
        """
    super_avg_max[subjectNumber].append(max_avg/len(result))
    max_sa[subjectNumber].append(sa_avg/len(result))
    userLabel.extend(ul)
    predictionLabel.extend(pl)
    return

def predictionFromList(subject_number,one_sign,model,model_count,mode):
    if mode == 0:
        tmp = all_csv_data[subject_number][one_sign]
    else:
        tmp = all_csv_data[model_count][one_sign]
    if GPU:
        tmp = cp.array(tmp)

    #split = np.array_split(tmp,15)
    split = np.array_split(tmp,60)
    result = []

    for data in split:
        tmp = model.predictor(Variable(cp.array(data)))
        result.extend(tmp.array.tolist())
    labeling(subject_number,result,model_count)
    return

"""
def unknownPredictionFromList(subject_number,one_sign,model,model_count):
    tmp = all_csv_data[model_count][one_sign]
    if GPU:
        tmp = cp.asarray(tmp)
    result = []
    split = np.array_split(tmp,15)
    for data in split:
        tmp = model.predictor(Variable(cp.array(data)))
        result.extend(tmp.array.tolist())
    labeling(subject_number,result,model_count)
    return
"""

def openPickel(file_path):
    with open(file_path, mode='rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

def main():
    global predictionLabel
    global userLabel
    global all_csv_data
    global all_csv_name
    global super_avg_max
    global max_sa
    global ten_mode
    global identification_mode
    global imode

    all_sign = 100 #1被験者の全署名の数
    samplecount = 1500
    """
    args = sys.argv
    if len(args) > 1:
        subfoldername = args[1]
        #samplecount = int(args[1])
    """
    #subfoldername = "x"
    #subfoldername = str(samplecount)

    filepath = "fcnnx/10/alltest"
    #filepath = "サンプル数別/" + subfoldername
    #被験者数設定
    subject_count = sum(1 for line in LoadSubjectList(f"{filepath}/model0.csv",0))-1
    word_count = 10
    #被験者リスト読み込み
    subjectnames = []
    for name in LoadSubjectList(f"{filepath}/model0.csv",1):
        subjectnames.append(name[0].decode("utf-8"))

    #Pickle全読み
    all_csv_data = [[list()]*all_sign for i in range(subject_count)] #全被験者の全署名
    all_csv_name = [[list()]*all_sign for i in range(subject_count)] #all_csv_dataのラベル(all_csv_name[0][0]はall_csv_data[0][0]のラベル)
    for i in range(len(all_csv_data)):
        all_csv_data[i] = openPickel(f'pickle/{samplecount}/Subject{i}_data.pickle')
        all_csv_name[i] = openPickel(f'pickle/{samplecount}/Subject{i}_sign.pickle')
    for ten in ten_mode:
        for mode in identification_mode:
            if mode == 0:
                capture_sign = all_sign - len(glob.glob(f"{filepath}/*.csv")) #署名の個数を指定した数取り出す
            else:
                capture_sign = all_sign
            if ten == True:
                capture_sign = 10

            #モデルごとでループ
            #for model_count in range(subject_count):
            for model_count in tqdm.tqdm(range(subject_count),desc="Model"):
                #モデル読み込み
                modeldata = f"{filepath}/test{model_count}.model"
                statedata = f"{filepath}/test{model_count}.state"
                model = L.Classifier(cnn_def.MLP(None, subject_count))
                if GPU:
                    chainer.cuda.get_device(GPU_ID).use()
                    model.to_gpu()
                optimizer = optimizers.Adam()
                optimizer.setup(model)
                chainer.serializers.load_npz(modeldata, model)
                chainer.serializers.load_npz(statedata, optimizer)

                #学習に利用した署名を格納
                learned_sign = [[] for i in range(subject_count)]
                tmp = [[] for i in range(len(glob.glob(f"{filepath}/*.csv")))]
                for i in range(len(tmp)):
                    for name in LoadSubjectList(f"{filepath}/model{i}.csv",1):
                        tmp[i].append(name[2+model_count].decode("utf-8"))
                for i in range(len(tmp)):
                    for j in range(subject_count):
                        learned_sign[j].append(tmp[i][j])

                #ランダムに検証するための処理(登録済みのサンプルは選択しない)
                all_random_index = [] #全被験者のランダムインデックス(2次元リスト)
                for i in range(subject_count):
                    random_index = list(range(len(all_csv_name[0])))
                    random.shuffle(random_index)
                    count = 0
                    tmp = []
                    #未登録者検証では全署名を利用
                    if mode == 1:
                        tmp = random_index
                    #登録者検証では登録済みの署名を選ばないようにする
                    else:
                        for j in random_index:
                            if count >= capture_sign:
                                break
                            if not all_csv_name[i][j] in learned_sign[i]:
                                tmp.append(j)
                                count += 1
                    all_random_index.append(tmp)

                fscore = [] #1モデルにおいてのすべての被験者のすべての署名のF値
                recall = []
                bcount = [] #1単語のサンプルが誰に何個分類されたかを記録
                #for one_sign in tqdm.tqdm(range(capture_sign),desc="Sign"):
                for one_sign in range(capture_sign):
                    predictionLabel = []
                    userLabel = []
                    for sub_num in range(subject_count):
                        if sub_num == model_count:
                            continue
                        else:
                            predictionFromList(sub_num,all_random_index[sub_num][one_sign],model,model_count,mode)
                    ul = np.array(userLabel,dtype=np.int32)
                    pl = np.array(predictionLabel,dtype=np.int32)
                    cm = confusion_matrix(ul,pl,labels=[x for x in range(subject_count)])
                    #print(cm)
                    #print(classification_report(ul,pl))
                    ul = np.split(ul,10)
                    pl = np.split(pl,10)
                    bcount2 = []
                    for i, splitpl in enumerate(pl):
                        
                        if i == model_count:
                            bcount2.append(99)
                        bcount2.append(np.argmax(np.bincount(splitpl)))
                    
                    if model_count == subject_count-1:
                        bcount2.append(99)
                    cs,cs2 = calculateScore(cm,subject_count,model_count)
                    bcount.append(bcount2)
                    fscore.append(cs)
                    recall.append(cs2)

                #print(f"Model:{model_count} complete.")
                if ten == True:
                    resultfilepath = f"{filepath}/identification/{mode}/{model_count}_"
                    os.makedirs(f"{filepath}/identification/{mode}/",exist_ok=True)
                else:
                    resultfilepath = f"{filepath}/threshould/{mode}/{model_count}_"
                    os.makedirs(f"{filepath}/threshould/{mode}/",exist_ok=True)
                #result = np.array(fscore,dtype=np.float32)
                np.savetxt(resultfilepath+"bincount.csv",bcount,delimiter=",", fmt='%d')

                for eval in eval_flag:
                    if eval == "fscore":
                        result = np.array(fscore)
                    else:
                        result = np.array(recall)
                    result = pd.DataFrame(data=result,columns=subjectnames)
                    result.to_csv(resultfilepath+eval+".csv",sep=",", index = False)
                    """
                    np.savetxt(resultfilepath+eval+".csv",fscore,fmt="%.6f",delimiter=",")
                    with open(resultfilepath+eval+".csv",'a')as f:
                        writer = csv.writer(f)
                        writer.writerow(subjectnames)
                        writer.writerow(["Average"])
                        writer.writerow(np.mean(result, axis = 0))
                        writer.writerow(["Standard deviation"])
                        writer.writerow(np.std(result, axis = 0))
                    """
            """
            resultfilepath = f"{filepath}/average/"
            os.makedirs(f"{filepath}/average/",exist_ok=True)
            sam = np.array(super_avg_max).T
            ms = np.array(max_sa).T
            np.savetxt(resultfilepath+f"{mode}_average.csv",sam,fmt="%.6f",delimiter=",")
            np.savetxt(resultfilepath+f"{mode}_sa.csv",ms,fmt="%.6f",delimiter=",")
            with open(resultfilepath+f"{mode}_average.csv",'a')as f:
                writer = csv.writer(f)
                writer.writerow(subjectnames)
                writer.writerow(["Average"])
                writer.writerow(np.mean(sam, axis = 0))
                writer.writerow(["Standard deviation"])
                writer.writerow(np.std(sam, axis = 0))
            with open(resultfilepath+f"{mode}_sa.csv",'a')as f:
                writer = csv.writer(f)
                writer.writerow(subjectnames)
                writer.writerow(["Average"])
                writer.writerow(np.mean(ms, axis = 0))
                writer.writerow(["Standard deviation"])
                writer.writerow(np.std(ms, axis = 0))
            """
            super_avg_max = [[],[],[],[],[],[],[],[],[],[],[]]
            max_sa = [[],[],[],[],[],[],[],[],[],[],[]]

    verification_mode = ["identification","threshould"]
    for mode in verification_mode:
        bc.bincount("rf",word_count,11,subject_count,mode)

if __name__ == '__main__':
    main()