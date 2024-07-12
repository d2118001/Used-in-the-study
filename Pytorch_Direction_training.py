#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import cupy as cp
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ExponentialLR
from pytorchtools import EarlyStopping
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
import Pytorch_cnn_1ch_nw_def as nw_def
#import igniter as ig
from matplotlib import pyplot as plt
import pandas as pd
from torchsummary import summary

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

def TrainBatch(train_data_size, train_loader, device, model, loss_func, optimizer):
    train_loss = 0
    train_acc = 0
    cnt = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_func(out, target)
        train_loss += loss.item()
        #_, pre = torch.max(out, 1)
        pre = out.max(1)[1]
        train_acc += (pre == target).sum().item()
        cnt += target.size(0)
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss / train_data_size
    avg_train_acc = train_acc / cnt
    return avg_train_loss, avg_train_acc

def ValBatch(val_data_size, val_loader, device, model, loss_func):
    val_loss = 0
    val_acc = 0
    cnt = 0
    model.eval()
    with torch.no_grad():  # 必要のない計算を停止
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_func(out, target)
            val_loss += loss.item()
            #_, pre = torch.max(out, 1)
            pre = out.max(1)[1]
            val_acc += (pre == target).sum().item()
            cnt += target.size(0)
    avg_val_loss = val_loss / val_data_size
    avg_val_acc = val_acc / cnt
    return avg_val_loss, avg_val_acc

def ViewGraph(epoch_num, train_loss_log, train_acc_log, val_loss_log, val_acc_log):
    plt.figure()
    plt.plot(range(epoch_num), train_loss_log, color="blue", linestyle="-", label="train_loss")
    plt.plot(range(epoch_num), val_loss_log, color="green", linestyle="--", label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and Validation loss")
    plt.grid()
    plt.savefig("log/Training and Validation loss.png")
    
    plt.figure()
    plt.plot(range(epoch_num), train_acc_log, color="blue", linestyle="-", label="train_acc")
    plt.plot(range(epoch_num), val_acc_log, color="green", linestyle="--", label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Training and Validation accuracy")
    plt.grid()
    plt.savefig("log/Training and Validation accuracy.png")
    #plt.show()

def main(data_dir,testCount):
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

    print('Leap motion Verification System by "Kamaishi"')
    print(f'GPU: {args.gpu}')
    print(f'# unit: {args.unit}')
    print(f'# Minibatch-size: {args.batchsize}')
    print(f'# epoch: {args.epoch}')
    print('')

    #forループ用
    torch.backends.cudnn.benchmark = True
    #chainer.config.train = True
    
    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    unitOut = len(glob.glob(data_dir+"*"))
    #model = L.Classifier(cnn_def.MLP(args.unit, unitOut))
    
    #GPU check
    if torch.cuda.is_available():
        d_type = "cuda"
    else:
        d_type = "cpu"
    device = torch.device(d_type)

    #Setup model
    model = nw_def.MLP(unitOut)
    model = model.to(device)
    summary(model,(1,72,20))
    """
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    """

    # Setup an optimizer
    #optimizer = chainer.optimizers.Adam()
    #optimizer.setup(model)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # -----データここから-----
    start = time.time()
    # テストデータの割合の分母
    trainDataRate = 3

    #Pickleを使う読み込み（速い）
    data_train_tmp = []
    label_train_tmp = []
    data_test_tmp = []
    label_test_tmp = []
    #出力ユニット数と同じ数(方向の数)繰り返す
    for i in range(unitOut):
        tmp = []
        #とりあえず1方向のサンプルを順番で格納していく(後でシャッフルする)
        for j in range(len(all_pickle_label[i])):
            tmp.append(all_pickle_data[i][j])
        random.shuffle(tmp) #サンプルの順番をシャッフルしてサンプルが偏らないようにする

        #trainDataRateでtmpを分割(t1,t2が学習用)
        t1,t2,t3 = np.array_split(np.array(tmp),trainDataRate)
        t1 = np.append(t1,t2,axis = 0) #t1,t2をくっつける
        data_test_tmp.extend(t3)
        data_train_tmp.extend(t1)

        #ラベル貼り
        for j in range(len(t1)):
            label_train_tmp.append(str(i))
        for j in range(len(t3)):
            label_test_tmp.append(str(i))

    elapsed_time = time.time() - start
    print (f"\nfile read time:{elapsed_time}[sec]")
    start = time.time()

    # データを4次元にしないと機械学習できないけどpickleでやってある(サンプル数,チャネル数,行,列)
    # 訓練,テストデータ(ラベルも)をPytorch用に変換
    train_data = np.array(data_train_tmp,dtype=np.float32)
    #train_data = np.squeeze(train_data)
    train_label = np.array(label_train_tmp,dtype=np.int32)

    test_data = np.array(data_test_tmp,dtype=np.float32)
    #test_data = np.squeeze(test_data)
    test_label = np.array(label_test_tmp,dtype=np.int32)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.int64)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.int64)

    """
    train = chainer.datasets.tuple_dataset.TupleDataset(train_data,train_label)
    test = chainer.datasets.tuple_dataset.TupleDataset(test_data,test_label)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize,repeat=True)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,repeat=False)
    """

    train_valid = torch.utils.data.TensorDataset(train_data,train_label)
    train_size = int(len(train_valid)*0.7)
    val_size = len(train_valid) - train_size
    train, valid = torch.utils.data.random_split(train_valid, [train_size, val_size])
    test =  torch.utils.data.TensorDataset(test_data,test_label)

    train_iter = torch.utils.data.DataLoader(train, batch_size=args.batchsize, shuffle=True)
    valid_iter = torch.utils.data.DataLoader(valid, batch_size=args.batchsize, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=args.batchsize, shuffle=False)

    # -----データここまで-----
    elapsed_time = time.time() - start
    print (f"data shape time:{elapsed_time}[sec]")
 
    # -----ここから学習-----
    start = time.time()

    epoch_num = args.epoch
    epoch_count = epoch_num

    early_stopping = EarlyStopping(patience=7, verbose=True)
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    print("epoch: ",epoch_num," batch size:", args.batchsize)
    print("start train.")
    for epoch in range(epoch_num):
        start_time = time.perf_counter()
        avg_train_loss, avg_train_acc = TrainBatch(len(train_iter.dataset),train_iter,device,model,loss_func,optimizer)
        end_time = time.perf_counter()

        s_val_time = time.perf_counter()
        avg_val_loss, avg_val_acc = ValBatch(len(valid_iter.dataset),valid_iter,device,model,loss_func)
        e_val_time = time.perf_counter()

        proc_time = end_time - start_time
        val_time = e_val_time - s_val_time
        print("Epoch[{}/{}], train loss: {loss:.4f}, valid loss: {val_loss:.4f}, valid acc: {val_acc:.4f}, "\
        "train time: {proc_time:.4f}sec, valid time: {val_time:.4f}sec"\
            .format(epoch+1, epoch_num, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc, 
            proc_time=proc_time, val_time=val_time))

        early_stopping(avg_val_loss, model) # 最良モデルならモデルパラメータを記録

        # 一定epochだけval_lossが最低値を更新しなかった場合、ここに入り学習を終了
        if early_stopping.early_stop:
            epoch_count = epoch
            break

        train_loss_log.append(avg_train_loss)
        train_acc_log.append(avg_train_acc)
        val_loss_log.append(avg_val_loss)
        val_acc_log.append(avg_val_acc)

    # モデルの保存
    torch.save(model.state_dict(), "log/mymodel.ckpt")
    print("save model.")

    #ViewGraph(epoch_count, train_loss_log, train_acc_log, val_loss_log, val_acc_log)
 
    """
    ig.run(
        epochs=args.epoch,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=ExponentialLR(optimizer, gamma=0.95),
        train_loader=train_iter,
        val_loader=valid_iter,
        device=device
    )
    """

    elapsed_time = time.time() - start
    print (f"training_time:{elapsed_time}[sec]")
    return test_iter
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
    chainer.serializers.load_npz('result/best.npz', trainer)
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

    #chainer.config.train = False
    
    nn.Module.eval()
    torch.backends.cudnn.benchmark = False

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

    #混同行列計算
    cm = confusion_matrix(test_label,prediction_label,labels=[i for i in range(unitOut)])
    calculateScore(cm,unitOut)

    #モデル,最適化情報保存
    chainer.serializers.save_npz('./Models/Direction.model', model)
    chainer.serializers.save_npz('./Models/Direction.state', optimizer)
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

def Test(data_dir,test_iter,testcount):
    unitOut = len(glob.glob(data_dir+"*"))
    
    torch.backends.cudnn.benchmark = False
    #fig = plt.figure()
    all_labels = np.array([])
    all_preds = np.array([])

    #データの読み込み
    test_loader = test_iter
    
    # select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    mymodel = nw_def.MLP(unitOut).to(device)
    mymodel.load_state_dict(torch.load(f"log/mymodel.ckpt"))

    mymodel.eval()
    with torch.no_grad():
        total = 0
        test_acc = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = mymodel(data)
            #_, preds = torch.max(outputs.data, 1)
            preds = outputs.max(1)[1]
            test_acc += (preds == target).sum().item()
            total += target.size(0)

            all_labels = np.append(all_labels, target.cpu().data.numpy())
            all_preds = np.append(all_preds, preds.cpu().numpy())

        print("正解率: {}%".format(100*test_acc/total))
        report = pd.DataFrame(classification_report(all_labels, all_preds, output_dict=True))
        report.to_csv(f"log/classification_report_{testcount}.csv")
        print(report)
        cm = confusion_matrix(all_labels, all_preds)
        #PlotConfusionMatrix(cm, np.array([i for i in range(unitOut)]))

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
def setSubjectCount(subjectCount):
    global allSubjectScore
    global allSubjecttpfpfntn
    for i in range(subjectCount):
        allSubjectScore.append([])
        allSubjecttpfpfntn.append([])

    global all_pickle_label
    global all_pickle_data
    for i in range(directioncount):
        all_pickle_data.append(openPickel(f'directionpickle/Direction{i}_data.pickle'))
        all_pickle_label.append(openPickel(f'directionpickle/Direction{i}_label.pickle'))
    #all_pickle_dataは[方向][サンプル1...サンプルx][チャネル数(1)][行][列]
    return

if __name__ == '__main__':
    testcount = 10
    path = "Direction/"
    #os.makedirs("./result/per_iteration/",exist_ok=True)
    os.makedirs("./result/per_subject/",exist_ok=True)
    os.makedirs("./log",exist_ok=True)

    files = os.listdir(path)
    directioncount = len(files)
    setSubjectCount(directioncount)
    allstart = time.time()
    for i in range(testcount):
        oneTest=path
        iter = main(oneTest,i)
        Test(oneTest,iter,i)
        #exit(0)
        #readLogJson(i)
    exit(0)
    for i in range(directioncount):
        fileWriter(allSubjecttpfpfntn[i],allSubjectScore[i],str(i))
    #oneFileWriter(iterationResult)

    allelapsed_time = time.time() - allstart
    print (f"elapsed_time:{allelapsed_time}[sec]")
    appendAveStdtoAlllog()
    os.system("dot_convert_to_png.bat")
