from __future__ import print_function
import argparse
import time
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda as xp
import cupy as cp
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
import allsignpickle as asp
import pickle

signcount = 100
row=20     #gyou
column=72   #retsu
GPU = True
GPU_ID = 0
chainer.config.train = False

def shaping(raw_data):
    data = np.asarray(raw_data,dtype=np.float32)
    data = np.reshape(data,(len(data),-1,row,column))
    return data

def main(subjectname,count,collectioncount,imitator):
    start = time.time()
    modeldata = "Models/Direction.model"
    statedata = "Models/Direction.state"
    nOut = 4

    model = L.Classifier(cnn_def.MLP(None, nOut))
    if GPU:
        chainer.cuda.get_device(GPU_ID).use()
        model.to_gpu()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    chainer.serializers.load_npz(modeldata, model, strict=False)
    chainer.serializers.load_npz(statedata, optimizer, strict=False)

    #File read
    start = time.time()
    path = f"imitatorpickle/{imitator}/{collectioncount}/"
    os.makedirs(path, exist_ok=True)

    all_csv_data = asp.openJoblibDump(f'mohousya_allsignpickle/{imitator}/{subjectname}_data.joblib')
    all_csv_name = asp.openJoblibDump(f'mohousya_allsignpickle/{imitator}/{subjectname}_sign.joblib')
    
    elapsed_time = time.time() - start
    #print ("File read elapsed time:{0}".format(elapsed_time) + "[sec]")

    #Classification
    splitcount = 300
    result = []
    for oneword in tqdm.tqdm(all_csv_data,desc=f"Split"):
        split = np.array_split(oneword,splitcount)
        tmp2 = []
        for data in split:
        #for data in tqdm.tqdm(split,desc=f"Sub{count}"):
            tmp = model.predictor(Variable(cp.array(data)))
            tmp2.extend(tmp.array.tolist())
        tmp2 = np.array(tmp2)
        result.append(tmp2)
    """
    for data in tqdm.tqdm(all_csv_data,desc=f"Sub{count}"):
        tmp = model.predictor(Variable(cp.array(data)))
        result.append(tmp.array.tolist())
    """
    start = time.time()
    #result = np.array(result)
    subjectresult = []
    for (i,r) in enumerate(result):
        resultmax = r.max(axis = 1) #１単語のデータの中から順番で4方向のうち一番シグモイドが大きかったものをリストに入れる
        resultmaxindex = r.argmax(axis = 1) #どの方向が一番大きかったか
        wordresult = []
        for j in range(collectioncount):
            wordresult.append(all_csv_data[i][resultmax.argmax()][0])  #iは何単語目かresultmax.argmax()は単語の中で一番シグモイドが大きかったデータ
            resultmax[resultmax.argmax()] = -1 #一番シグモイドが大きかったデータを消す(消すわけではない)
        wordresult = shaping(wordresult)
        subjectresult.append(wordresult)

    elapsed_time = time.time() - start
    #print ("Classification elapsed time:{0}".format(elapsed_time) + "[sec]")
    model.to_cpu()
    #File write
    start = time.time()
    del all_csv_data
    del result
    del tmp2
    del tmp

    with open(f'{path}Subject{count}_data.pickle', mode='wb') as f:
        pickle.dump(subjectresult,f)
    with open(f'{path}Subject{count}_sign.pickle', mode='wb') as f:
        pickle.dump(all_csv_name,f)

    elapsed_time = time.time() - start
    #print ("File write elapsed time:{0}".format(elapsed_time) + "[sec]")
    return

if __name__ == '__main__':
    args = sys.argv

    #Single subject mode
    if len(args) > 2:
        subjectname = args[1]
        subjectnumber = args[2]
        collectioncount = int(args[3])
        main(subjectname,subjectnumber,collectioncount)
    #Multi subject mode
    elif len(args) > 1:
        subjectname = ["a","b","c","d"]
        print("Subjects Count:"+str(len(subjectname)))
        collectioncount = int(args[1])
        imitator = "ia"
        for (i,name) in enumerate(subjectname):
            if os.path.exists(f'imitatorpickle/{imitator}/{collectioncount}/Subject{i}_data.pickle') == False:
                main(name,i,collectioncount,imitator)
            else:
                print(f'imitatorpickle/{imitator}/{collectioncount}/Subject{i}_data.pickle is already exist')