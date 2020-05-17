import pandas as pd
import math
import string
import numpy as np

inputfile_dir = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\data\\train.csv"
outputfile1 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\train_raw1.csv"
outputfile2 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Topic_data\\train_raw1.csv"
df = pd.read_csv(inputfile_dir, header=None)
nTweets = len(df)
newdata1 = []
newdata2 = []

def data2df(newdata):
    columns = ['text','response']
    data = [
        [newdata[i][0],newdata[i][1]]
        for i in range(len(newdata))]
    df2 = pd.DataFrame(data, columns=columns)
    return df2

for i in range(nTweets):
    if df.iloc[i][1]=="0" or df.iloc[i][1]=="1":
        newdata1.append(df.iloc[i])
    elif df.iloc[i][1]=="2":
        newdata1.append((df.iloc[i][0],"0"))

dr1 = data2df(newdata1)
dr1.to_csv(outputfile1, mode='a', index=False, header=False)

for i in range(nTweets):
    if df.iloc[i][1]=="3":
        newdata2.append((df.iloc[i][0],1))
    else:
        newdata2.append((df.iloc[i][0],0))

dr2 = data2df(newdata2)
dr2.to_csv(outputfile2, mode='a', index=False, header=False)