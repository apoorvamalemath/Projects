import numpy as np
import pandas as pd

df = pd.read_csv('../distance.txt', header = None, sep=" ")

no_rows=df.shape[0]
for i in range(no_rows):
	rowdf=df.iloc[i]
	rowdf=rowdf.sort_values(ascending=True)
	query="RQ"+format(i+1, "04")
	f = open(query, "w")
	f.write("%s\n" % query)	
	for j in range(len(rowdf)-1):
		dfile="D"+format((rowdf.index[j])+1, "05")
		f.write("%s "% dfile)
		f.write("%s\n"%rowdf[rowdf.index[j]])

