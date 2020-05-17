import os
import pandas as pd
import numpy as np


"""inputfile_dir = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\test1.csv"
outputfile = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\test.csv"

tweets_pd = pd.read_csv(inputfile_dir)
nTweets = len(tweets_pd)
newdata = []
for i in range(nTweets):
    str = tweets_pd.iloc[i][0]
    str1 = str.replace(',',"")
    print(str1)
    newdata.append((str1,tweets_pd.iloc[i][1]))

df1 = pd.DataFrame(newdata, columns = tweets_pd.columns)
df1.to_csv(outputfile, index=False)"""

inputfile_dir = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\test.csv"
inputfile_dir2 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\train.csv"
outputfile = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\train_raw.csv"
df1 = pd.read_csv(inputfile_dir)
df2 = pd.read_csv(inputfile_dir2)
nTweets = len(df1)
nTweets2 = len(df2)
newdata = []
for i in range(nTweets):
    str = df1.iloc[i][0]
    newdata.append((str,1-int(df1.iloc[i][1])))
for i in range(nTweets2):
    str = df2.iloc[i][0]
    newdata.append((str,1-int(df2.iloc[i][1])))
df3 = pd.DataFrame(newdata, columns=df1.columns)
df3.to_csv(outputfile, index=False)