import pandas as pd
import pickle
import csv
import numpy as np
from operator import itemgetter
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tweet_obj import tweet_obj
from basic_analysis import *
from LSI_analyzer import *
from feature_extraction import *
from decomposition_analysis import *
from lp_optimizer import *
from basic_analysis import Topics, TopicsNames, filter_set
from w2v_analyzer import w2v_analyzer



dir = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\"

'''
read data
'''
input_path_old = dir + "atrain1.csv"

#replace special word
tweets_pd = pd.read_csv(input_path_old)
tweets_pd = tweets_pd.replace(r'\\n',' ', regex=True)
tweets_pd = tweets_pd.replace(r"xef",' ', regex=True)
tweets_pd = tweets_pd.replace(r'covid19','coronavirus', regex=True)
tweets_pd = tweets_pd.replace(r'covid','coronavirus', regex=True)
tweets_pd = tweets_pd.replace(r'covid-19','coronavirus', regex=True)
tweets_pd = tweets_pd.replace(r'corona virus','coronavirus', regex=True)
tweets_pd = tweets_pd.replace(r'\.\'',' ', regex=True)
tweets_pd = tweets_pd.replace(r'xbb',' ', regex=True)
nTweets = len(tweets_pd)
nMyTopics = len(TopicsNames)
topicSimi = 0.7
k = 3.0

# analysis

def getNormalTokens(tweetObject):
    normalTokens = list()
    for tokenObject in tweetObject.tokenList:
        normalTokens.append(tokenObject.content)
    return normalTokens


def list2string(tokenList):
    s = ''
    for i, token in enumerate(tokenList):
        s = s + token
        if i < len(tokenList)-1:
            s = s + ' '
    return s

# analyze all tweets, get their tokens with their types
normalWord = list()
textList = list()
tokensList = list()
texts = list()
for i in range(nTweets):
    tweetObj = tweet_obj(tweets_pd.iloc[i])
    tweetObj.text_analyzer
    normalTokens = getNormalTokens(tweetObj)
    normalWord.extend(normalTokens)
    textList.append(tweetObj.getClearText())
    tokensList.append(normalTokens)
    texts.append(list2string(normalTokens))

print("------------------Basic Analysis------------------")
print('%d tweets, %d words totally.' % (nTweets, len(normalWord)))

tweets_pd['normalText'] = texts
basic_analysis(normalWord)

# topics analysis
nTopics = nMyTopics
print("------------------------------------------")
print('LSI:')
lsi_analyzer = LSI_analyzer(nTopics)
X_Lsi = lsi_analyzer.fit_transform(texts)
topics_words, topics_probs = lsi_analyzer.get_topics(7)
i = 0
for topic_words, topic_probs in zip(topics_words, topics_probs):
    i += 1
    print("Topic" + str(i))
    print(topic_words)


LSIpickle_path = dir+ 'LSIdata_' + str(nTopics) + 'topics.pkl'
LSIpickle_file = open(LSIpickle_path, 'wb')
pickle.dump((X_Lsi, lsi_analyzer, topics_words, topics_probs), LSIpickle_file)
LSIpickle_file.close()

# LDA
def cluster_analysis(X, nTopics):
    from sklearn.cluster import KMeans
    cluster = KMeans(n_clusters=nTopics, random_state=0)
    cluster.fit(X)
    return cluster.labels_, cluster.cluster_centers_

def clusterIdAnalysis(Xclus, Cluster, maxC, minC):
    n, nX = Xclus.shape
    nCluster = Cluster.shape[0]
    assert Cluster.shape[1] == nX
    XCdist = list()
    for i in range(n):
        for j in range(nCluster):
            XCdist.append((i, j, np.linalg.norm(Xclus[i] - Cluster[j], ord=2)))
    XCdist.sort(key=itemgetter(2))
    cluster = Counter(range(nCluster))
    for j in range(nCluster):
        cluster[j] = 0
    clusterMatrix = -1 * np.ones(n, dtype=int)

    for tup in XCdist:
        i, j, dist = tup
        if clusterMatrix[i] < 0 and cluster[j] < minC:
            clusterMatrix[i] = j
            cluster[j] += 1

    for tup in XCdist:
        i, j, dist = tup
        if clusterMatrix[i] < 0 and cluster[j] < maxC:
            clusterMatrix[i] = j
            cluster[j] += 1
    assert np.min(clusterMatrix) >= 0
    return clusterMatrix


def feature_extraction(texts, min_ngram, max_ngram, modelType):
    max_df = 1.0
    min_df = 0
    maxFeature = 500
    if modelType == 'tf':
        extractor = CountVectorizer(analyzer='word', ngram_range=(min_ngram, max_ngram), max_features=maxFeature, encoding='utf-8', strip_accents='unicode', stop_words=filter_set, max_df=max_df, min_df=min_df)
        X = extractor.fit_transform(texts)
        featureNames = extractor.get_feature_names()
    elif modelType == 'tf-idf':
        extractor = TfidfVectorizer(analyzer='word', ngram_range=(min_ngram, max_ngram), max_features=maxFeature, encoding='utf-8', strip_accents='unicode', stop_words=filter_set, max_df=max_df, min_df=min_df)
        X = extractor.fit_transform(texts)
        featureNames = extractor.get_feature_names()

    else:
        return None, None, None

    return extractor, X, featureNames


def decomposition_analysis(X, nTopics, featureNames=None):
    learning_offset = 20
    learning_method = 'batch'
    analyzer = LatentDirichletAllocation(n_components=nTopics, max_iter=50, learning_method=learning_method, learning_offset=learning_offset, random_state=0)
    X_trans = analyzer.fit_transform(X)
    for topic_idx, topic in enumerate(analyzer.components_):
        print("Topic%d" % topic_idx)
        print(" ".join([str(topic[i]) + '*' + featureNames[i] for i in topic.argsort()[:20 - 1:-1]]))
    return X_trans, analyzer

cluster_LDA = True
nCluster = 10
# cluster analysis
print('Cluster analysis:')
valid_irow = [i for i in range(nTweets)]
Xclus = X_Lsi[valid_irow]
clusterId, clusterFeature = cluster_analysis(Xclus, nCluster)
MinCluster = np.floor(len(valid_irow)/float(nCluster))
MaxCluster = np.ceil(len(valid_irow)/float(nCluster))
clusterIdMatrix = clusterIdAnalysis(Xclus, clusterFeature, MaxCluster, MinCluster)
# label the cluster id for each tweet
count = 0
clusterids = list()
for irow, row in tweets_pd.iterrows():
    clusterids.append(clusterIdMatrix[count])
    count += 1
tweets_pd['clusterId'] = clusterids
print(Counter(tweets_pd['clusterId'].values))
groupText = list()
for iCluster in range(nCluster):
    clusterText = [row['normalText'] for irow, row in tweets_pd.iterrows() if row['clusterId'] == iCluster]
    gt = " ".join(clusterText)
    groupText.append(gt)

# feature extraction and decomposition
print('Feature extraction:')
modelType = 'tf-idf'  # model type: tf, tf-idf
extractor_LDAex, X_LDAex, featureNames_LDAex = feature_extraction(groupText, 1, 1, modelType)
print('Decomposition analysis:')
methodType = 'LDA'
X_trans, analyzer = decomposition_analysis(X_LDAex, nTopics, featureNames_LDAex)



# show LDA graphically
showLDAPic = False
if methodType == 'LDA' and showLDAPic:
    import pyLDAvis, pyLDAvis.sklearn
    # pyLDAvis.enable_notebook()
    data_pyLDAvis = pyLDAvis.sklearn.prepare(analyzer, X_LDAex, extractor_LDAex)
    pyLDAvis.show(data_pyLDAvis)


# calculate similarities between tweets and topic keywords
def transFun(x, M, k):
    if x < 0:
        return 0.0
    if x < 0.01:
        return 1.0
    assert x <= M
    y = 1.0 - 2.0/(1 + np.exp(k*(M/x-1)))
    return y

def tweet2df(tweets):
    columns = ['text']
    data = [
        [tweet['text']]
        for tweet in tweets]
    df = pd.DataFrame(data, columns=columns)
    return df

w2vModelSource = 'GoogleNews'
w2vMmethod = 1  # methodType: 0: CBOW; 1: skip-gram
w2vAnalyzer = w2v_analyzer(w2vModelSource, w2vMmethod, 300, 40, filter_set)
for i, theTopic in enumerate(Topics):
    for keyword in theTopic:
        try:
            wv = w2vAnalyzer.model.wv[keyword]
        except KeyError:
            print('In Topic' + str(i + 1) + ', keyword: ' + keyword + ' not found in w2v vocabulary!')


w2vTokenCount = list()
for text in texts:
    wvs, d = w2vAnalyzer.getSenMat(text.split())
    w2vTokenCount.append(d.shape[0])
tweets_pd['w2vTokensCount'] = w2vTokenCount

for Topic, topicName in zip(Topics, TopicsNames):
    TopicDiff = list()
    for text in texts:
        diff = w2vAnalyzer.getSenDiff(text.split(), Topic, 1)
        TopicDiff.append(diff)
    tweets_pd[topicName] = TopicDiff
simi = tweets_pd[TopicsNames].values
M = np.max(simi)*1.0001
simiMatrix = np.zeros((nTweets, len(Topics)+1))
for i in range(nTweets):
    min = 0
    for j in range(len(Topics)):
        simiMatrix[i, j] = transFun(simi[i, j], M, k)
        if(simiMatrix[i, j]>simiMatrix[i,min]):
            min = j
    if(simiMatrix[i,min]>topicSimi):
        simiMatrix[i,len(Topics)]=min
    else:
        simiMatrix[i, len(Topics)] =-1


for j, theTopic in enumerate(Topics):
    tweets_pd[TopicsNames[j]+'_simi'] = simiMatrix[:, j]

tweets_pd['topicid'] = simiMatrix[:, len(Topics)]
output_filename = "rumor_tweets_analyzed.csv"
output_path = dir +  output_filename
tweets_pd.to_csv(output_path, index=False)

simi = tweets_pd[TopicsNames].values
simi_trans = tweets_pd[[topicName+'_simi' for topicName in TopicsNames]].values
print(simi)

# initial output
sipping_water = []
hold_breathe = []
bath = []
mosquiton = []
antibiotics = []
salt_water = []
garlic_water = []
cold_weather = []
hot_and_humid_weather= []


input_path = dir + "rumor_tweets_analyzed.csv"
tweets_pdall = pd.read_csv(input_path)
numTweets = len(tweets_pdall)
for i in range(nTweets):
    if(tweets_pdall.iloc[i]['topicid']>=0):
        if(tweets_pdall.iloc[i]['topicid']==0):
            sipping_water.append(tweets_pdall.iloc[i])
        elif(tweets_pdall.iloc[i]['topicid']==1):
            hold_breathe.append(tweets_pdall.iloc[i])
        elif(tweets_pdall.iloc[i]['topicid'] == 2):
            bath.append(tweets_pdall.iloc[i])
        elif(tweets_pdall.iloc[i]['topicid'] == 3):
            mosquiton.append(tweets_pdall.iloc[i])
        elif (tweets_pdall.iloc[i]['topicid'] == 4):
            antibiotics.append(tweets_pdall.iloc[i])
        elif (tweets_pdall.iloc[i]['topicid'] == 5):
            salt_water.append(tweets_pdall.iloc[i])
        elif (tweets_pdall.iloc[i]['topicid'] == 6):
            garlic_water.append(tweets_pdall.iloc[i])
        elif (tweets_pdall.iloc[i]['topicid'] == 7):
            cold_weather.append(tweets_pdall.iloc[i])
        elif (tweets_pdall.iloc[i]['topicid'] == 8):
            hot_and_humid_weather.append(tweets_pdall.iloc[i])
sipping_water_df = tweet2df(sipping_water)
sipping_water_df = sipping_water_df['text']
sipping_water_df.to_csv(dir + "sipping_water_df.csv", index=False)
hold_breathe_df = tweet2df(hold_breathe)
hold_breathe_df.to_csv(dir + "hold_breathe_df.csv", index=False)
bath_df = tweet2df(bath)
bath_df.to_csv(dir + "bath_df.csv", index=False)
mosquito_df = tweet2df(mosquiton)
mosquito_df.to_csv(dir + "mosquiton_df.csv", index=False)
antibiotics_df = tweet2df(antibiotics)
antibiotics_df.to_csv(dir + "antibiotics_df.csv", index=False)
salt_water_df = tweet2df(salt_water)
salt_water_df.to_csv(dir + "salt_water_df.csv", index=False)
garlic_water_df = tweet2df(garlic_water)
garlic_water_df.to_csv(dir + "garlic_water_df.csv", index=False)
cold_weather_df = tweet2df(cold_weather)
cold_weather_df.to_csv(dir + "cold_weather_df.csv", index=False)
hot_and_humid_weather_df = tweet2df(hot_and_humid_weather)
hot_and_humid_weather_df.to_csv(dir + "hot_and_humid_weather_df.csv", index=False)