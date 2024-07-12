import pickle
import csv
import tqdm
import sys
import os
import glob
import concurrent.futures
import numpy as np

row=0      #gyou
column=0   #retsu

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
    return data

def openPickel(file_path):
    with open(file_path, mode='rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

def main():
    all_csv_data = []
    all_csv_label = []
    datadir = "Direction/"    
    directionlist = [os.path.basename(r) for r in glob.glob(datadir+"*")]
    for (i,direction) in enumerate(directionlist):
        csvfiles = [os.path.basename(r) for r in glob.glob(datadir+direction+"/*")]
        for csvfile in csvfiles:
            tmp = load_data(datadir+direction+"/"+csvfile)
            all_csv_data.append(tmp)
            all_csv_label.append(direction)
        with open(f'directionpickle/Direction{i}_data.pickle', mode='wb') as f:
            pickle.dump(shaping(all_csv_data),f)
        with open(f'directionpickle/Direction{i}_label.pickle', mode='wb') as f:
            pickle.dump(all_csv_label,f)
        all_csv_data = []
        all_csv_label = []

def readTest():
    datadir = "Direction/"    
    directionlist = [os.path.basename(r) for r in glob.glob(datadir+"*")]
    for i in range(len(directionlist)):
        all_csv_data = openPickel(f'directionpickle/Direction{i}_data.pickle')
        all_csv_label = openPickel(f'directionpickle/Direction{i}_label.pickle')
        print(all_csv_data[-1][-1][-1][-1]) #最後尾

if __name__ == '__main__':
    main()
    readTest()
