import string
import pandas as pd
import numpy as np
import re
import ast
import csv

def cleanx():
    inputfile_dir = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\antibiotics_df1.csv"
    outputfile1 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\clean hot.csv"
    inputfile_dir2 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\antibiotics_df.csv"
    outputfile = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\final_anti.csv"
    df = pd.read_csv(inputfile_dir, header=None)
    nTweets = len(df)
    newdata = []
    newdata.append("text")
    for i in range(1,nTweets):
        line = df.iloc[i][0][2:-1]
        tmp = list(line)
        length = len(tmp)
        for i in range(length):
            if tmp[i]=='\\' and tmp[i+1]=='x':
                tmp[i:i+4] = ['','','','']
        tmp1 = ''.join(tmp)
        newdata.append(tmp1)

    df1 = pd.DataFrame(newdata, columns = df.columns)
    df1.to_csv(outputfile1, mode='a', index=False, header=False)

    df2 = pd.read_csv(outputfile1, header=None)
    df3 = pd.read_csv(inputfile_dir2, header=None, skiprows=1)
    df4 = pd.DataFrame(np.concatenate([df2.values, df3.values]), columns = df2.columns)
    df4.to_csv(outputfile, mode='a', index=False, header=False)

def cleanURL(inputfile_dir,outputfile):
    df = pd.read_csv(inputfile_dir, header=None)
    url_reg = r'[a-z]*[:.]+\S+'
    nTweets = len(df)
    newdata = []
    for i in range(nTweets):
        str = df.iloc[i][0]
        result = re.sub(url_reg, '', str)
        print(result)
        newdata.append((result, df.iloc[i][1]))
    df1 = pd.DataFrame(newdata, columns=['text', 'label'])
    df1.to_csv(outputfile, index=False)

def cleanUSER(inputfile_dir,outputfile):
    df = pd.read_csv(inputfile_dir, header=None)
    User = r'@+\S+'
    nTweets = len(df)
    newdata = []
    for i in range(nTweets):
        str = df.iloc[i][0]
        result = re.sub(User, '', str)
        print(result)
        newdata.append((result, df.iloc[i][1]))
    df1 = pd.DataFrame(newdata, columns=['text', 'label'])
    df1.to_csv(outputfile, index=False)

def cleanTag(inputfile_dir,outputfile):
    df = pd.read_csv(inputfile_dir, header=None)
    Tag = r'#+\S+'
    nTweets = len(df)
    newdata = []
    for i in range(nTweets):
        str = df.iloc[i][0]
        result = re.sub(Tag, '', str)
        print(result)
        newdata.append((result,df.iloc[i][1]))
    df1 = pd.DataFrame(newdata, columns=['text','label'])
    df1.to_csv(outputfile, index=False)



if __name__ == '__main__':
    inputfile_dir = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\train_21.csv"
    outputfile = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\train_2.csv"
    #cleanx()
    #cleanURL(inputfile_dir,outputfile)
    cleanUSER(inputfile_dir,outputfile)
    #cleanTag(inputfile_dir,outputfile)

