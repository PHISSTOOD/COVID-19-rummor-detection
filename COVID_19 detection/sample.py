import pandas as pd
import math
import string
import numpy as np

inputfile_dir = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\train_raw1.csv"
outputfile1 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\train_22.csv"
outputfile2 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\test_22.csv"

df = pd.read_csv(inputfile_dir, header=None)
df1 = df.sample(frac=0.9, replace=False, random_state=1, axis=0)
newdata = []
newdata1 = []
nTweets1 = len(df1)
nTweets2 = len(df)

def data2df(newdata):
    columns = ['text','response']
    data = [
        [newdata[i][0],newdata[i][1]]
        for i in range(len(newdata))]
    ds = pd.DataFrame(data, columns=columns)
    return ds


df = df.append(df1).drop_duplicates(keep=False)

df1.to_csv(outputfile1, mode='a', index=False, header=False)
df.to_csv(outputfile2, mode='a', index=False, header=False)