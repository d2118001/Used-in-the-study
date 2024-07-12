from hashlib import sha1
from pprint import pprint
import numpy as np
import sys
import os

def main():
	#bincount("複数単語",10,4,4)
	#mode = "identification"
	mode = "threshould"
	#bincount("cnnx",10,11,11,mode)
	bincount("victimfcnn",10,5,10,mode)
def bincount(dir_name,word_count,model_count,subject_count,mode):
	for i in range(1,2):
		all_sign = np.zeros((model_count,subject_count),dtype = int)
		dpath = f"{dir_name}/{word_count}/alltest/{mode}/{i}"
		os.makedirs(f"{dpath}/bincount",exist_ok=True)
		for j in range(model_count):
			loadcsv = np.loadtxt(f"{dpath}/{j}_bincount.csv",delimiter = ',',dtype=int).T
			#loadcsv = np.delete(loadcsv,j,0)
			all_bcount = []
			for c in loadcsv:
				bcount = np.bincount(c)
				while len(bcount) < subject_count:
					bcount = np.append(bcount,0)
				all_bcount.append(bcount.tolist())
			#all_bcount.insert(j,([0] * subject_count))
			#print(np.shape(all_bcount))
			np.savetxt(f"{dpath}/bincount/model{j}bincount.csv",all_bcount,fmt="%d",delimiter=",")

			for bcount in all_bcount:
				all_sign += bcount
			np.savetxt(f"{dpath}/bincount/allmodelbincount.csv",all_sign,fmt="%d",delimiter=",")

if __name__ == '__main__':
    main()