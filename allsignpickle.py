import pickle
import joblib
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

def loadCsvToList(file_path,subject_number,one_sign,sign_name):
    global all_csv_data
    global all_csv_name
    tmp = []
    for fp in tqdm.tqdm(file_path,desc=f"Sub:{subject_number}Sig:{one_sign}"):
        tmp.append(load_data(fp))
    all_csv_data[one_sign] = shaping(tmp)
    all_csv_name[one_sign] = sign_name
    return

def shaping(raw_data):
    data = np.asarray(raw_data,dtype=np.float32)
    data = np.reshape(data,(len(data),-1,row,column))
    return data

def openJoblibDump(file_path):
    with open(file_path, mode='rb') as f:
        joblibdump = joblib.load(f)
    return joblibdump

def openPickel(file_path):
    with open(file_path, mode='rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

#本人用
def main(subjectnames,all_sign):
    global all_csv_data
    global all_csv_name
    #CSV全読み
    for i in tqdm.tqdm(range(len(subjectnames)),desc="Subject"):
        #配列の初期値の0は読み込んだCSVのデータの配列とラベルの文字列で上書きされる
        all_csv_data = [0] * all_sign       #1被験者の全署名
        all_csv_name = ["0"] * all_sign     #all_csv_dataのラベル(all_csv_name[0]はall_csv_data[0]のラベル)
        file_path = subjectnames[i]+"/"
        name = subjectnames[i]
        files = os.listdir(name)
        signs = [f for f in files if os.path.isdir(os.path.join(name, f))]
        with concurrent.futures.ThreadPoolExecutor(max_workers = 10) as executor:
            for j in range(all_sign):
                executor.submit(loadCsvToList,glob.glob(f"{file_path}{signs[j]}/20/*"),i,j,signs[j])
        #並列処理したときに正常に読み込めてたかチェック
        for j in tqdm.tqdm(range(all_sign),desc=f"Check"):
            #ラベルのリストの要素が初期値のまま=並列処理時に実行されなかった
            if all_csv_name[j] == "0":
                loadCsvToList(glob.glob(f"{file_path}{signs[j]}/20/*"),i,j,signs[j])
            #並列処理した結果が正しく順番通りかチェック
            if all_csv_name[j] != signs[j]:
                print(f"\nFAIL:{all_csv_name[j]},{signs[j]}\n")

        #pickleだとファイルサイズが大きくなりすぎるのでjoblibを使う
        with open(f'C:\\Signs\\allsignpickle\\{subjectnames[i]}_data.joblib', mode='wb') as f: 
            joblib.dump(all_csv_data, f, compress=3)
        with open(f'C:\\Signs\\allsignpickle\\{subjectnames[i]}_sign.joblib', mode='wb') as f:
            joblib.dump(all_csv_name, f, compress=3)

#なりすまし攻撃者用
def main(subjectnames,all_sign,impersonator,impersonator_path):
    global all_csv_data
    global all_csv_name
    #CSV全読み
    for i in tqdm.tqdm(range(len(subjectnames)),desc="Subject"):
        #配列の初期値の0は読み込んだCSVのデータの配列とラベルの文字列で上書きされる
        all_csv_data = [0] * all_sign       #1被験者の全署名
        all_csv_name = ["0"] * all_sign     #all_csv_dataのラベル(all_csv_name[0]はall_csv_data[0]のラベル)
        file_path = impersonator_path+subjectnames[i]+"/"
        name = impersonator_path+subjectnames[i]
        files = os.listdir(name)
        signs = [f for f in files if os.path.isdir(os.path.join(name, f))]
        with concurrent.futures.ThreadPoolExecutor(max_workers = 10) as executor:
            for j in range(all_sign):
                executor.submit(loadCsvToList,glob.glob(f"{file_path}{signs[j]}/20/*"),i,j,signs[j])
        #並列処理したときに正常に読み込めてたかチェック
        for j in tqdm.tqdm(range(all_sign),desc=f"Check"):
            #ラベルのリストの要素が初期値のまま=並列処理時に実行されなかった
            if all_csv_name[j] == "0":
                loadCsvToList(glob.glob(f"{file_path}{signs[j]}/20/*"),i,j,signs[j])
            #並列処理した結果が正しく順番通りかチェック
            if all_csv_name[j] != signs[j]:
                print(f"\nFAIL:{all_csv_name[j]},{signs[j]}\n")

        #pickleだとファイルサイズが大きくなりすぎるのでjoblibを使う
        with open(f'C:\\Signs\\mohousya_allsignpickle\\{impersonator}\\{subjectnames[i]}_data.joblib', mode='wb') as f: 
            joblib.dump(all_csv_data, f, compress=3)
        with open(f'C:\\Signs\\mohousya_allsignpickle\\{impersonator}\\{subjectnames[i]}_sign.joblib', mode='wb') as f:
            joblib.dump(all_csv_name, f, compress=3)

#読み込みテスト
def readTest(subjectnames,all_sign):
    for i in tqdm.tqdm(range(len(subjectnames))):
        all_csv_data = openJoblibDump(f'C:\\Signs\\allsignpickle\\{subjectnames[i]}_data.joblib')
        all_csv_name = openJoblibDump(f'C:\\Signs\\allsignpickle\\{subjectnames[i]}_sign.joblib')
        print(all_csv_name)

def readTest(subjectnames,all_sign,impersonator):
    for i in tqdm.tqdm(range(len(subjectnames))):
        all_csv_data = openJoblibDump(f'C:\\Signs\\mohousya_allsignpickle\\{impersonator}\\{subjectnames[i]}_data.joblib')
        all_csv_name = openJoblibDump(f'C:\\Signs\\mohousya_allsignpickle\\{impersonator}\\{subjectnames[i]}_sign.joblib')
        print(all_csv_name)

if __name__ == '__main__':
    impersonator = "subjectname"
    impersonator_path = "imitater/subjectname(imitater)/"
    
    os.makedirs(f"C:\\Signs\\mohousya_allsignpickle\\{impersonator}\\", exist_ok=True)
    subjectnames = ["a","b","c"]
    all_sign = 100 #1被験者の全署名の数
    
    main(subjectnames,all_sign,impersonator,impersonator_path)
    readTest(subjectnames,all_sign,impersonator)
    
    #main(subjectnames,all_sign)
    #readTest(subjectnames,all_sign)