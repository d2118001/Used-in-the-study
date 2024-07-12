from __future__ import print_function
import argparse
from pprint import pprint
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
import glob
import cupy as cp
import concurrent.futures
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as prfs
from traitlets import import_item
import imitatorbincounter as bc
import cnn_1ch_nw_def as cnn_def
#import fc_1ch_nw_def as cnn_def

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

def calculateScore(cm,subject_count):
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

def labeling(subjectNumber,result):
    global userLabel
    global predictionLabel
    result = np.array(result)
    result = F.softmax(result).array
    #result = F.sigmoid(result).array
    ul = []
    pl = []
    #result.data1行(1つのサンプル)ごとでsoftmaxが最大値の被験者をplにいれる
    for signResult in result:
        ul.append(subjectNumber)
        #pl.append(np.argmax(signResult))
        #if subjectNumber == np.argmax(signResult) and np.max(signResult) >= 0.9:
        pl.append(np.argmax(signResult))
        """
        if subjectNumber == np.argmax(signResult):
            pl.append(np.argmax(signResult))
        else:
            pl.append(model_count)
        """
    userLabel.extend(ul)
    predictionLabel.extend(pl)
    return

def rf_labeling(subjectNumber,result):
    global userLabel
    global predictionLabel
    global super_avg_max
    global max_sa
    result = np.array(result)
    predictionLabel.extend(result)

    ul = []
    for signResult in result:
        ul.append(subjectNumber)
    userLabel.extend(ul)
    return

def predictionFromList(subject_number,one_sign,model):
    tmp = all_csv_data[subject_number][one_sign]
    if GPU:
        tmp = cp.array(tmp)

    #split = np.array_split(tmp,15)
    split = np.array_split(tmp,60)
    result = []
    for data in split:
        tmp = model.predictor(Variable(cp.array(data)))
        result.extend(tmp.array.tolist())
    labeling(subject_number,result)
    return

def rf_predictionFromList(subject_number,one_sign,model):
    tmp = all_csv_data[subject_number][one_sign]
    test_data = np.reshape(tmp,(len(tmp),1440))
    result = model.predict(test_data)
    rf_labeling(subject_number,result)
    return

def openPickel(file_path):
    with open(file_path, mode='rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

#いじるのはこの中だけ
#---------------------------------------------------------------------------------
imitators=["a","b","c","d","e"]
resultpath = "victimrf"
all_sign = 100 #1被験者の全署名の数
samplecount = 1500

#resultpath = "victimfcnn"
filepathvictim = f"{resultpath}/10/alltest"
filepathpickle = f"pickle_and_victimpickle/{samplecount}"

#被験者数設定
subject_count = 10

#使用する単語の数
capture_sign = 100

#ニューラルネットを使うか機械学習を使うか
nn = False
#--------------------------------------------------------------------------------
def main(imitator, imitator_num):
    global predictionLabel
    global userLabel
    global all_csv_data
    global all_csv_name

    filepathimitator = f"imitatorpickle/{imitator}/{samplecount}/"
    imitatorpicklelist = glob.glob(filepathimitator+"/*_data.pickle")
    imitatorpicklenamelist = glob.glob(filepathimitator+"/*_sign.pickle")
    #Pickle全読み
    all_csv_data = [[list()]*all_sign for i in range(subject_count)] #全被験者の全署名
    all_csv_name = [[list()]*all_sign for i in range(subject_count)] #all_csv_dataのラベル(all_csv_name[0][0]はall_csv_data[0][0]のラベル)

    #被験者リスト読み込み
    subjectnames = []
    for i in range(subject_count):
        subjectnames.append(str(i))

    for i in range(len(imitatorpicklenamelist)):
        all_csv_name[i] = openPickel(imitatorpicklenamelist[i])
        all_csv_data[i] = openPickel(imitatorpicklelist[i])
        if nn == False:
            all_csv_data[i] = np.squeeze(all_csv_data[i])
    i+=1
    picklelen = len(glob.glob(filepathpickle+"/*_data.pickle"))
    for j in range(i, picklelen):
        all_csv_data[j] = openPickel(filepathpickle+f"/Subject{j}_data.pickle")
        all_csv_name[j] = openPickel(filepathpickle+f"/Subject{j}_sign.pickle")
        if nn == False:
            all_csv_data[j] = np.squeeze(all_csv_data[j])

    #ランダムフォレストかニューラルネットワークか
    if nn == False:
        with open(f"{filepathvictim}/model_{0}.sav","rb") as f:
            model = pickle.load(f)
    else:
        modeldata = f"{filepathvictim}/test{0}.model"
        statedata = f"{filepathvictim}/test{0}.state"
        model = L.Classifier(cnn_def.MLP(None, subject_count))
        if GPU :
            chainer.cuda.get_device(GPU_ID).use()
            model.to_gpu()
        optimizer = optimizers.Adam()
        optimizer.setup(model)
        chainer.serializers.load_npz(modeldata, model)
        chainer.serializers.load_npz(statedata, optimizer)

    all_random_index = [] #全被験者のランダムインデックス(2次元リスト)
    for i in range(subject_count):
        random_index = list(range(len(all_csv_name[0])))
        random.shuffle(random_index)
        all_random_index.append(random_index)

    fscore = [] #1モデルにおいてのすべての被験者のすべての署名のF値
    recall = []
    bcount = [] #1単語のサンプルが誰に何個分類されたかを記録
    for one_sign in tqdm.tqdm(range(capture_sign),desc="Sign"):
    #for one_sign in range(capture_sign):
        predictionLabel = []
        userLabel = []
        for sub_num in range(subject_count):
            if nn == False:
                rf_predictionFromList(sub_num,all_random_index[sub_num][one_sign],model)
            else:
                predictionFromList(sub_num,all_random_index[sub_num][one_sign],model)
        ul = np.array(userLabel,dtype=np.int32)
        pl = np.array(predictionLabel,dtype=np.int32)
        cm = confusion_matrix(ul,pl,labels=[0,1,2,3,4,5,6,7,8,9])
        #print(cm)
        #print(classification_report(ul,pl))
        ul = np.split(ul,10)
        pl = np.split(pl,10)

        bcount2 = []
        for i, splitpl in enumerate(pl):
            bcount2.append(np.argmax(np.bincount(splitpl)))
        cs,cs2 = calculateScore(cm,subject_count)
        bcount.append(bcount2)
        fscore.append(cs)
        recall.append(cs2)

    #print(f"Model:{model_count} complete.")
    resultfilepath = f"{filepathvictim}/threshould/{1}/{imitator_num}_"
    os.makedirs(f"{filepathvictim}/threshould/{1}/",exist_ok=True)
    #result = np.array(fscore,dtype=np.float32)
    np.savetxt(resultfilepath+"bincount.csv",bcount,delimiter=",", fmt='%d')

    for eval in eval_flag:
        if eval == "fscore":
            result = np.array(fscore)
        else:
            result = np.array(recall)
        result = pd.DataFrame(data=result,columns=subjectnames)
        result.to_csv(resultfilepath+eval+".csv",sep=",", index = False)

    return

if __name__ == '__main__':
    for i,imitator in enumerate(imitators):
        main(imitator,i)
    bc.bincount(resultpath,10,len(imitators),10,"threshould")
    