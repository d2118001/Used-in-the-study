import numpy as np
import sys
import os

def main():
	#bincount("複数単語",10,4,4)
	mode = "identification"
	#mode = "threshould"
	bincount("cnnx",10,11,11,mode)
def bincount(dir_name,word_count,model_count,subject_count,mode):
	for i in range(2):
		all_sign = np.zeros((subject_count,model_count),dtype = int)
		dpath = f"{dir_name}/{word_count}/alltest/{mode}/{i}"
		#dpath = f"サンプル数別/{x}/identification/{i}"
		os.makedirs(f"{dpath}/bincount",exist_ok=True)
		for j in range(model_count):
			loadcsv = np.loadtxt(f"{dpath}/{j}_bincount.csv",delimiter = ',',dtype=int).T
			loadcsv = np.delete(loadcsv,j,0)
			all_bcount = []
			for c in loadcsv:
				bcount = np.bincount(c)
				while len(bcount) < subject_count:
					bcount = np.append(bcount,0)
				#print(bcount)
				all_bcount.append(bcount.tolist())
			all_bcount.insert(j,([0] * subject_count))
			#print(np.shape(all_bcount))
			np.savetxt(f"{dpath}/bincount/model{j}bincount.csv",all_bcount,fmt="%d",delimiter=",")
			all_sign += all_bcount
			np.savetxt(f"{dpath}/bincount/allmodelbincount.csv",all_sign,fmt="%d",delimiter=",")

if __name__ == '__main__':
    main()