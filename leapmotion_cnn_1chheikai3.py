#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
from pprint import pprint
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
import shutil
import tqdm
import concurrent.futures
import pickle
import cupy as cp
import numpy as np
from friday13 import readLogJson
from chainer import training,Variable
from chainer.training import extensions
from chainer.training import triggers
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import confusion_matrix
import cnn_1ch_nw_def as cnn_def
#import fc_1ch_nw_def as cnn_def
import pandas as pd
row=0      #gyou
column=0   #retsu
iterationResult = []
allSubjecttpfpfntn = []
allSubjectScore = []
all_train_word = []
all_pickle_data = []
all_pickle_name = []
word_count = 0 #local_word_countから変更すること

def load_data(dataPath):
    global row
    global column
    loadedText = np.loadtxt(dataPath,delimiter = ',')
    column = len(loadedText[0])
    row = sum(1 for line in open(dataPath))
    return np.array(loadedText,dtype=np.float32)

def shaping(raw_data):
    data = np.asarray(raw_data,dtype=np.float32)
    data = np.reshape(data,(len(data),-1,row,column))
    cp.asarray(data)
    return data

def openPickel(file_path):
    with open(file_path, mode='rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

def calculateScore(cm,subjectCount,testCount):
    testCount = int(testCount)
    print(cm)
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
        if i == testCount:
            pre = 0
            rec = 0
            fs = 0
            fr = 0
            fa = 0
        else:
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
        """
        if i == testCount:
            continue
        """
        allSubjecttpfpfntn[i].append(np.array([tp[i],tn[i],fp[i],fn[i]],dtype=np.int32))
        allSubjectScore[i].append(np.array([precision[i],recall[i],fscore[i],far[i],frr[i],eer[i]],dtype=np.float32))
    return

def main(testCount,label_list,result_path,subject_name,train_sign_path):
    parser = argparse.ArgumentParser(description='Chainer example: Leap motion by Kamaishi')
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
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

    print('Leap motion Verification System by "Kamaishi"')
    print(f'# GPU: {args.gpu}')
    print(f'# unit: {args.unit}')
    print(f'# Minibatch-size: {args.batchsize}')
    print(f'# epoch: {args.epoch}')

    #forループ用
    chainer.config.train = True

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    unitOut = len(subject_name)
    #unitOut = len(glob.glob(data_dir+"/*"))
    model = L.Classifier(cnn_def.MLP(args.unit, unitOut))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # -----データここから-----
    # データ読み込みと整形
    # テストデータの割合の分母
    trainDataRate = 3
    start = time.time()
    #テストデータ,訓練データ,ラベル作成
    data_train_tmp = []
    label_train_tmp = []
    data_valid_tmp = []
    data_test_tmp = []
    label_test_tmp = []
    label_valid_tmp = []

    #学習に利用した単語
    train_word = [[] for i in range(word_count)]
    for i in range(word_count):
        train_word[i].append(testCount)

    #被験者の数だけループ
    for i in range(len(subject_name)):
        #モデル番号と被験者番号が同じ場合はその被験者は登録しない
        """
        if int(testCount) == i:
            for i in range(word_count):
                train_word[i].append("")
            continue
        """
        count = 0 #1被験者の何文字目か
        #100文字からランダムでword_count分取り出す
        for one_sign in random.sample(all_pickle_name[i],word_count):
            train_word[count].append(os.path.basename(one_sign))
            tmp =  all_pickle_data[i][all_pickle_name[i].index(one_sign)]
            random.shuffle(tmp) #配列をシャッフルして1文字内で学習とテストでデータが偏らないようにする
            #trainDataRateに分割
            t1,t2,t3 = np.split(tmp,trainDataRate)
            t1 = np.append(t1,t2,axis = 0) #t1,t2をくっつける
            data_train_tmp.extend(t1)
            #data_valid_tmp.extend(t2)
            data_test_tmp.extend(t3)
            for j in range(len(t1)):
                label_train_tmp.append(str(i))
                #label_valid_tmp.append(str(i))
            for j in range(len(t3)):
                label_test_tmp.append(str(i))
            count += 1
    
    for i in range(word_count):
        all_train_word[i].append(train_word[i])
    elapsed_time = time.time() - start
    print (f"\nfile read time:{elapsed_time}[sec]")
    start = time.time()
    # データを4次元にしないと機械学習できないけどpickleでやってある(サンプル数,チャネル数,行,列)
    # 訓練,テストデータ(ラベルも)をchainer用に変換
    train_data = np.array(data_train_tmp,dtype=np.float32)
    train_label = np.array(label_train_tmp,dtype=np.int32)

    #valid_data = np.array(data_valid_tmp,dtype=np.float32)
    #valid_label = np.array(label_valid_tmp,dtype=np.int32)

    test_data = np.array(data_test_tmp,dtype=np.float32)
    test_label = np.array(label_test_tmp,dtype=np.int32)

    """
    print(np.shape(train_data))
    print(np.shape(train_label))
    print(np.shape(valid_data))
    print(np.shape(valid_label))
    print(np.shape(test_data))
    print(np.shape(test_label))
    """

    train = chainer.datasets.tuple_dataset.TupleDataset(train_data,train_label)
    #valid = chainer.datasets.tuple_dataset.TupleDataset(valid_data,valid_label)
    test = chainer.datasets.tuple_dataset.TupleDataset(test_data,test_label)

    #これがわかればk-foldValidationが簡単に実装できるはず
    """
    dataset = list(zip(train_data, train_label))
    train = chainer.datasets.get_cross_validation_datasets(dataset,10)
    """
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize,repeat=True)
    #valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,repeat=False)

    # -----データここまで-----
    del data_test_tmp
    del data_train_tmp
    elapsed_time = time.time() - start
    print (f"data shape time:{elapsed_time}[sec]")

    start = time.time()
    # -----ここから学習-----
    # Set up a trainer

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    stop_trigger = triggers.EarlyStoppingTrigger(monitor='validation/main/loss',max_trigger=(args.epoch, 'epoch'), patients=7)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)
    trainer.extend(
        chainer.training.extensions.snapshot(filename='best.npz'),
        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss'))

    #trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

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
    print (f"training time:{elapsed_time}[sec]")
    #-----結果出力ここから-----

    start = time.time()
    prediction_label = []

    #推定された答え(精度など出力)←MemoryErrorで死にやすい
    """
    model.to_cpu()
    p = model.predictor(Variable(test_data))
    softmax = F.softmax(p)
    for i in range(len(test_data)):
        prediction_label.append(np.argmax(softmax.data[i]))
    acc = F.accuracy(p,Variable(test_label))
    print("accuracy:"+str(acc.data))
    cross = F.softmax_cross_entropy(p,Variable(test_label))
    print("cross entropy:"+str(cross.data))
    """

    #訓練モードOFF(しないとDropoutとBatch Normalizationで結果が死ぬ)
    chainer.config.train = False
    #速い推定(GPUにやらせる)
    p = []
    for t in test_data:
        tmp = model.predictor(Variable(cp.array([t])))
        p.append(tmp[0].array.tolist())
    p =np.array(p)

    model.to_cpu()

    softmax = F.softmax(p)
    for i in range(len(test_data)):
        prediction_label.append(np.argmax(softmax.data[i]))
    acc = F.accuracy(p,Variable(test_label))
    print("Accuracy:"+str(acc.data))
    cross = F.softmax_cross_entropy(p,Variable(test_label))
    print("Cross entropy:"+str(cross.data))

    """
    #推定された答え(MemoryError対策版)
    model.to_cpu()
    for one_data in test_data:
        #1個ずつPredictorに入れる
        one_data = np.array([one_data])
        p = model.predictor(Variable(one_data))
        softmax = F.softmax(p)
        prediction_label.append(np.argmax(softmax.data))
    """

    #分類結果出力
    print(classification_report(test_label, prediction_label))

    report = classification_report(test_label, prediction_label, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(f"{result_path}/{int(testCount)}_report.csv")

    #混同行列計算
    cm = confusion_matrix(test_label,prediction_label,labels=[i for i in range(len(subject_name))])
    calculateScore(cm,unitOut,testCount)

    #モデル,最適化情報保存
    chainer.serializers.save_npz("./"+train_sign_path+"/test"+testCount+".model", model)
    chainer.serializers.save_npz("./"+train_sign_path+"/test"+testCount+".state", optimizer)
    print("モデル保存完了")

    elapsed_time = time.time() - start
    print (f"calculate result time:{elapsed_time}[sec]")
    #-----結果出力ここまで-----

    #?
    """
    #TP,TN,FP,FN計算+データ保存
    tp = []
    tn = []
    fp = []
    fn = []
    tmp = np.sum(cm,axis=0)
    for j in range(unitOut):
        tp.append(cm[j][j])
        fp.append(np.sum(cm[j])-tp[j])
        fn.append(tmp[j]-tp[j])
    for j in range(unitOut):
        tn.append(np.sum(tp)-tp[j])
    path = "./hoge/per_iteration/Iteration"+str(testCount)
    np.savetxt(path+"_confusion_matrix.txt",cm,fmt="%.6f",delimiter="\t")
    with open(path+'_tp_tn_fp_fn.csv','a')as f:
        writer = csv.writer(f)
        writer.writerow(tp)
        writer.writerow(tn)
        writer.writerow(fp)
        writer.writerow(fn)
    ef = open(path+".txt","w")
    ef.write("accuracy:"+str(acc.data)+"\n")
    result = prfs(test_label, prediction_label)
    np.savetxt(path+"_precision_recall_fscore_support.txt",result,fmt="%.6f",delimiter="\t")
    #print(prfs(test_label, prediction_label))

    #複数回の学習の平均
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
    ef.write(classification_report(test_label, prediction_label))
    ef.close()
    """

    return
    """
    じんせいくるしいたけ
      ／⌒＼ 
      (> <)
      ) ͡　! 
     （__ノ
    """

def oneFileWriter(result,result_path):
    result = np.array(result,dtype=np.float32)
    filePath = "./"+result_path+"/per_iteration/"
    np.savetxt(filePath+"test.csv",result,fmt="%.6f",delimiter=",")
    print(result)
    print(np.mean(result, axis = 0))
    with open(filePath+"test.csv",'a')as f:
        writer = csv.writer(f)
        writer.writerow(["Precision","Recall","F-score","Acuraccy"])
        writer.writerow(["Average"])
        writer.writerow(np.mean(result, axis = 0))
        writer.writerow(["Standard deviation"])
        writer.writerow(np.std(result, axis = 0))
    global iterationResult
    iterationResult = []
    return

def fileWriter(result1,result2,subjectNumber,result_path):
    filePath = "./"+result_path+"/per_subject/"+subjectNumber+"_"
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

def setGlobalVariable(subjectCount,wc,subject_name,all_sign,samplecount,pickle_path):
    global word_count
    word_count = wc
    global allSubjectScore
    global allSubjecttpfpfntn
    for i in range(subjectCount):
        allSubjectScore.append([])
        allSubjecttpfpfntn.append([])
    global all_train_word
    for i in range(word_count):
        all_train_word.append([])
    for i in range(word_count):
        tmp = []
        tmp.append("Subject name")
        for j in subject_name:
            tmp.append(os.path.basename(j))
        all_train_word[i].append(tmp)
        tmp = []
        tmp.append("Subject number")
        for j in range(subjectCount):
            tmp.append(str(j))
        all_train_word[i].append(tmp)
    #Pickle全読み
    global all_pickle_name
    global all_pickle_data
    all_pickle_data = [[list()]*all_sign for i in range(len(subject_name))] #全被験者の全署名(all_pickle_data[0][0]は被験者1の1番目のデータ)
    all_pickle_name = [[list()]*all_sign for i in range(len(subject_name))] #all_pickle_dataのラベル(all_pickle_name[0][0]はall_pickle_data[0][0]のラベル)
    for i in range(len(all_pickle_data)):
        all_pickle_data[i] = openPickel(f'{pickle_path}/{samplecount}/Subject{i}_data.pickle')
        #shaping(all_pickle_data[i])
        all_pickle_name[i] = openPickel(f'{pickle_path}/{samplecount}/Subject{i}_sign.pickle')
    return

if __name__ == '__main__':

    result_path = "victimresultCNN"
    pickle_path = "pickle_and_victimpickle"
    test_type = "victimcnn"

    all_sign = 100          #一人当たりの単語は何個?
    local_word_count = 10   #学習時に何単語使うか
    samplecount = 1500      #サンプル数いくつ?

    train_sign_path = f"{test_type}/{local_word_count}/alltest/"

    subject_name = ["a","b","c","d","e","f","g","h","i","j"]

    #os.makedirs("./"+result_path+"/per_iteration/",exist_ok=True)
    #os.makedirs("./"+result_path+"/per_subject/",exist_ok=True)
    os.makedirs(train_sign_path,exist_ok=True)
    os.makedirs(result_path,exist_ok=True)

    label_list = [str(i) for i in range(len(subject_name))]
    print(label_list)

    setGlobalVariable(len(label_list),local_word_count,subject_name,all_sign,samplecount,pickle_path)
    allstart = time.time()

    a = ["0"]
    for i in a:
    #for i in label_list:
        main(i,label_list,result_path,subject_name,train_sign_path)
        readLogJson(int(i))
    #oneFileWriter(iterationResult)
    
    for i in a:
    #for i in range(len(all_train_word)):
        with open(f"{train_sign_path}model{str(i)}.csv","w",newline='') as f:
            writer = csv.writer(f)
            all_train_word[i] = np.array(all_train_word[i]).T
            for j in all_train_word[i]:
                writer.writerow(j)
    
    #for i in range(len(label_list)):
    #    fileWriter(allSubjecttpfpfntn[i],allSubjectScore[i],str(i),result_path)
    
    allelapsed_time = time.time() - allstart
    print (f"allelapsed_time:{allelapsed_time}[sec]")
    #os.system("dot_convert_to_png.bat")