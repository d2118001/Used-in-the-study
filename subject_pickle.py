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
all_csv_data = []
all_csv_name = []

def load_data(dataPath):
    global row
    global column
    loadedText = np.loadtxt(dataPath,delimiter = ',')
    column = len(loadedText[0])
    row = sum(1 for line in open(dataPath))
    return np.array(loadedText,dtype=np.float32)

def LoadSubjectList(data_path,skipRow):
    return np.array(np.loadtxt(data_path,delimiter = ',',dtype=bytes,skiprows=skipRow))

def loadCsvToList(file_path,subject_number,one_sign,sign_name):
    global all_csv_data
    global all_csv_name
    tmp = []
    for fp in tqdm.tqdm(file_path,desc=f"Sub:{subject_number}Sig:{one_sign}"):
    #for fp in file_path:
        tmp.append(load_data(fp))
    all_csv_data[subject_number][one_sign] = shaping(tmp)
    all_csv_name[subject_number][one_sign] = sign_name
    return

def shaping(raw_data):
    data = np.asarray(raw_data,dtype=np.float32)
    data = np.reshape(data,(len(data),-1,row,column))
    return data

def openPickel(file_path):
    with open(file_path, mode='rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

def main():
    global all_csv_data
    global all_csv_name
    all_sign = 100 #1被験者の全署名の数

    #被験者数設定
    subject_count = sum(1 for line in LoadSubjectList("alltest/0subject.csv",0))-1
    #被験者リスト読み込み
    subjectnames = []
    for name in LoadSubjectList("./alltest/0subject.csv",1):
        subjectnames.append(name[0].decode("utf-8"))

    #CSV全読み
    all_csv_data = [[list()]*all_sign for i in range(subject_count)] #全被験者の全署名
    all_csv_name = [[list()]*all_sign for i in range(subject_count)] #all_csv_dataのラベル(all_csv_name[0][0]はall_csv_data[0][0]のラベル)
    for i in tqdm.tqdm(range(subject_count),desc="Subject"):
        file_path = "./disassembled/"+subjectnames[i]+"/"
        signs = [os.path.basename(r) for r in glob.glob(file_path+"*")]
        with concurrent.futures.ThreadPoolExecutor(max_workers = 20) as executor:
            for j in range(all_sign):
                executor.submit(loadCsvToList,glob.glob(f"{file_path}{signs[j]}/20/*"),i,j,signs[j])
        for j in tqdm.tqdm(range(all_sign),desc=f"check:{j}"):
            if len(all_csv_data[i][j]) == 0:
                loadCsvToList(glob.glob(f"{file_path}{signs[j]}/20/*"),i,j,signs[j])
            if all_csv_name[i][j] != signs[j]:
                print(f"\nFAIL:{all_csv_name[i][j]},{signs[j]}\n")
        with open(f'hoge/Subject_test{i}_data.pickle', mode='wb') as f:
        	pickle.dump(all_csv_data[i],f)
        with open(f'hoge/Subject_test{i}_sign.pickle', mode='wb') as f:
            pickle.dump(all_csv_name[i],f)
            
    #読み込みテスト
    for i in range(len(all_csv_data)):
        all_csv_data[i] = openPickel(f'pickle/Subject{i}_data.pickle')
        all_csv_name[i] = openPickel(f'pickle/Subject{i}_sign.pickle')
        print(all_csv_name[i])

def test():
    global all_csv_data
    global all_csv_name
    all_sign = 100
    subject_count = sum(1 for line in LoadSubjectList("alltest/0subject.csv",0))-1

    all_csv_data = [[list()]*all_sign for i in range(subject_count)] #全被験者の全署名
    all_csv_name = [[list()]*all_sign for i in range(subject_count)] #all_csv_dataのラベル(all_csv_name[0][0]はall_csv_data[0][0]のラベル)

    for i in range(len(all_csv_data)):
        all_csv_data[i] = openPickel(f'pickle/Subject{i}_data.pickle')
        all_csv_name[i] = openPickel(f'pickle/Subject{i}_sign.pickle')
    print(len(all_csv_data[0][0][0][0][0]))

if __name__ == '__main__':
    #test()
    main()