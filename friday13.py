#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import csv
import numpy as np

def loadData(datapath):
    loadedText = np.loadtxt(datapath, delimiter = ',', skiprows = 1)
    return np.array(loadedText,dtype=np.float32)

def readLogJson(testcount = 0):
	#学習1回ごとのCSV
	with open(f"./result/log{testcount}.csv","w",newline='') as f:
		writer = csv.writer(f)
		writer.writerow(["epoch","main/loss","main/accuracy",
			"validation/main/loss","validation/main/accuracy",
			"iteration","elapsed_time"])
	result = json.load(open("./result/log","r"))
	for epoch in result:
		with open(f"./result/log{testcount}.csv","a",newline='') as f:
			writer = csv.writer(f)
			writer.writerow([epoch["epoch"],epoch["main/loss"],epoch["main/accuracy"],
				epoch["validation/main/loss"],epoch["validation/main/accuracy"],
				epoch["iteration"],epoch["elapsed_time"]])

	#すべての学習の最後のエポックをまとめたCSV(上のforループが終わった後の変数epochを使う)
	if testcount == 0:
		with open("./result/alllog.csv","w",newline='') as f:
			writer = csv.writer(f)
			writer.writerow(["epoch","main/loss","main/accuracy",
				"validation/main/loss","validation/main/accuracy",
				"iteration","elapsed_time"])
	with open("./result/alllog.csv","a",newline='') as f:
		writer = csv.writer(f)
		writer.writerow([epoch["epoch"],epoch["main/loss"],epoch["main/accuracy"],
			epoch["validation/main/loss"],epoch["validation/main/accuracy"],
			epoch["iteration"],epoch["elapsed_time"]])
	return

#alllogの後ろに平均と標準偏差をつける(readLogJsonが全テスト回数分実行された後に実行すること)
def appendAveStdtoAlllog():
	alllog = loadData("./result/alllog.csv")
	with open("./result/alllog.csv","a",newline='') as f:
		writer = csv.writer(f)
		writer.writerow(["Average"])
		writer.writerow(np.mean(alllog, axis = 0))
		writer.writerow(["Standard deviation"])
		writer.writerow(np.std(alllog, axis = 0))

if __name__ == '__main__':
	print("このプログラムは単品で実行しないでください")
