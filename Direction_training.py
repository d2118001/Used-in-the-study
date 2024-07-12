#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time
import chainer
import chainer.functions as F
import chainer.links as L
import csv
import numpy as np
import os
import sys
import random
import glob
import pickle
import cupy as cp
from chainer import training,Variable
from friday13 import readLogJson
from friday13 import appendAveStdtoAlllog
from chainer.training import extensions
from chainer.training import triggers
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import confusion_matrix
import pandas as pd
import cnn_1ch_nw_def as cnn_def
#import fc_1ch_nw_def as cnn_def
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

def main(testCount, result_path, direction_count):
    parser = argparse.ArgumentParser(description='Chainer example: Leap motion by Kamaishi')
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

    #print('Leap motion NNtrain System by "Kamaishi"')
    print(f'GPU: {args.gpu} unit: {args.unit} Minibatch-size: {args.batchsize} epoch: {args.epoch}')

    #forループ用
    chainer.config.train = True

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    unitOut = direction_count
    model = L.Classifier(cnn_def.MLP(args.unit, unitOut))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # -----データここから-----

    start = time.time()
    # データ読み込みと整形
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

    #特徴量を減らす(列を少なくする)
    """
    data_test_tmp2 = []
    data_train_tmp2 = []
    for tmp in data_test_tmp:
        tmp2 = tmp[0]
        tmp3 = tmp2[:,:54]
        data_test_tmp2.append(tmp3)
    for tmp in data_train_tmp:
        tmp2 = tmp[0]
        tmp3 = tmp2[:,:54]
        data_train_tmp2.append(tmp3)

    data_test_tmp2 = np.reshape(data_test_tmp2,(len(data_test_tmp),-1,20,54))
    data_train_tmp2 = np.reshape(data_train_tmp2,(len(data_train_tmp2),-1,20,54))

    data_test_tmp2 =np.asarray(data_test_tmp2,dtype=np.float32)
    data_train_tmp2 =np.asarray(data_train_tmp2,dtype=np.float32)

    print(np.shape(data_test_tmp2))
    print(np.shape(data_train_tmp2))
    """

    elapsed_time = time.time() - start
    print (f"\nfile read time:{elapsed_time}[sec]")
    start = time.time()

    # データを4次元にしないと機械学習できないけどpickleでやってある(サンプル数,チャネル数,行,列)
    # 訓練,テストデータ(ラベルも)をchainer用に変換
    #train_data = np.array(data_train_tmp2,dtype=np.float32)
    train_data = np.array(data_train_tmp,dtype=np.float32)
    train_label = np.array(label_train_tmp,dtype=np.int32)
    #test_data = np.array(data_test_tmp2,dtype=np.float32)
    test_data = np.array(data_test_tmp,dtype=np.float32)
    test_label = np.array(label_test_tmp,dtype=np.int32)

    valid_data = np.array(data_valid_tmp,dtype=np.float32)
    valid_label = np.array(label_valid_tmp,dtype=np.int32)

    train = chainer.datasets.tuple_dataset.TupleDataset(train_data,train_label)
    valid = chainer.datasets.tuple_dataset.TupleDataset(valid_data,valid_label)
    test = chainer.datasets.tuple_dataset.TupleDataset(test_data,test_label)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize,repeat=True)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,repeat=False)
    
    # -----データここまで-----

    elapsed_time = time.time() - start
    print (f"data shape time:{elapsed_time}[sec]")

    start = time.time()
    # -----ここから学習-----
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)    

    # Use Early stopping
    stop_trigger = triggers.EarlyStoppingTrigger(monitor='validation/main/loss',max_trigger=(args.epoch, 'epoch'), patients=3)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)
    trainer.extend(
        chainer.training.extensions.snapshot(filename='best.npz'),
        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss'))

    #No use Early stopping
    #trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    #trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    #trainer.extend(extensions.snapshot_object(model, 'model_snapshot_{.updater.epoch}', trigger=(1,'epoch')))
    #trainer.extend(extensions.snapshot_object(model, 'model_snapshot_{.updater.epoch}'))
    
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))

    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy','elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()
    #chainer.serializers.load_npz('result/best.npz', trainer)
    # -----ここまで学習-----
    elapsed_time = time.time() - start
    print (f"training_time:{elapsed_time}[sec]")
    #-----結果出力ここから-----
    # # print(len(train_data[0]))
    # p = model.predictor(Variable(np.asarray([test_data[0]],dtype=np.float32)))
    # for i in range(len(test_data)):
    #     print("-----")
    #     print(F.softmax(p).data)
    #     print(test_label[i])

    start = time.time()

    chainer.config.train = False

    #速い推定(GPUにやらせる)
    p = []

    for t in test_data:
        tmp = model.predictor(Variable(cp.array([t])))
        p.append(tmp[0].array.tolist())
    p =np.array(p)

    model.to_cpu()

    #遅い推定(メモリーエラーにならない)
    #p = model.predictor(Variable(test_data))

    #推定の答え合わせ
    softmax = F.softmax(p)
    prediction_label = []
    for i in range(len(test_data)):
        prediction_label.append(np.argmax(softmax.data[i]))
        """
        print("-----")
        print(softmax.data[i])
        print(test_label[i])
        """
    #精度など出力
    acc = F.accuracy(p,Variable(test_label))
    print("Accuracy:"+str(acc.data))
    cross = F.softmax_cross_entropy(p,Variable(test_label))
    print("Cross entropy:"+str(cross.data))
    print(classification_report(test_label, prediction_label))


    print(classification_report(test_label, prediction_label))

    report = classification_report(test_label, prediction_label, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(f"{result_path}/{int(testCount)}_report.csv")

    #混同行列計算
    cm = confusion_matrix(test_label,prediction_label,labels=[i for i in range(unitOut)])
    calculateScore(cm,unitOut)

    #モデル,最適化情報保存
    #chainer.serializers.save_npz('./Models/Direction.model', model)
    #chainer.serializers.save_npz('./Models/Direction.state', optimizer)
    print("モデル保存完了")
    elapsed_time = time.time() - start
    print (f"calculate result time:{elapsed_time}[sec]")

    """
    result = prfs(test_label, prediction_label)
    resultAve = np.empty(0,dtype=np.float32)
    global iterationResult
    count = 0
    for element in result:
        if count < 3:
            resultAve = np.append(resultAve,np.average(element))
        count+=1
    resultAve = np.append(resultAve,acc.data)
    #print(resultAve)
    iterationResult.append(resultAve)
    """
    return

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
    pickle_path = "directionpickle"
    result_path = "directionresultCNN"
    os.makedirs(result_path, exist_ok=True)

    directioncount = 4
    setSubjectCount(directioncount, pickle_path)
    allstart = time.time()
    for i in range(testcount):
        main(i, result_path, directioncount)
        readLogJson(i)

    """
    for i in range(directioncount):
        fileWriter(allSubjecttpfpfntn[i],allSubjectScore[i],str(i))
    #oneFileWriter(iterationResult)
    """

    allelapsed_time = time.time() - allstart
    print (f"allelapsed_time:{allelapsed_time}[sec]")
    appendAveStdtoAlllog()
    #os.system("dot_convert_to_png.bat")