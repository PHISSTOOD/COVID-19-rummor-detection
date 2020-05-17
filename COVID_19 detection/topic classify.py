import pandas as pd
import math
import string
import numpy as np

inputfile_dir = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\data\\Data_raw.csv"
outputfile = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\data\\train.csv"
df = pd.read_csv(inputfile_dir, header=None)
nTweets = len(df)
newdata = []
numlist = []

def data2df(newdata,numlist):
    columns = ['text','response']
    data = [
        [newdata[i],numlist[i]]
        for i in range(len(newdata))]
    df1 = pd.DataFrame(data, columns=columns)
    return df1

def trans(st):
    if st == "Exposing rumors":
        return 0
    elif st == "Supporting rumors":
        return 1
    elif st == "Neither of first two categories(but related to the topic)":
        return 2
    else:
        return 3

for i in range(math.floor(nTweets/3)):
    count = [0,0,0,0]
    newdata.append(df.iloc[i*3+1][27])
    str1 = df.iloc[i*3 + 0][28]
    str2 = df.iloc[i*3 + 1][28]
    str3 = df.iloc[i*3 + 2][28]
    num1 = trans(str1)
    num2 = trans(str2)
    num3 = trans(str3)
    count[num1] = count[num1]+1
    count[num2] = count[num2] + 1
    count[num3] = count[num3] + 1
    if(count[0]>=2):
        numlist.append(0)
    elif(count[1]>=2):
        numlist.append(1)
    elif (count[2] >= 2):
        numlist.append(2)
    elif(count[3]>=2):
        numlist.append(3)
    elif(count[3]==0):
        numlist.append(2)
    else:
        numlist.append(3)

dr = data2df(newdata,numlist)
dr.to_csv(outputfile, mode='a', index=False, header=False)