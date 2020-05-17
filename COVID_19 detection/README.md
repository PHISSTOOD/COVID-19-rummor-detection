COVID_19 Misinformation Dtection
==========
## Introduction
Health problems have troubled more and more people. Among them, the problems of obesity, hypertension and hyperlipidemia are often mentioned. Therefore, for specific people, the doctor will recommend the amount of oil, salt and sugar to be ingested every day. But many people will feel very hungry and uncomfortable after not eating enough food. Therefore, this project aim to build a small model that can help some people eat more food and make sure they do not exceed the limitation.
## Programming Language:
Python 3
## Python Package requirment:
numpy, pandas, sklearn, genSim, NLTK, pyLDAvis, tweepy, tensorflow, etc.
## Other package requirement:
word2vec package: Google news word2vec package, you can download it from https://github.com/mmihaltz/word2vec-GoogleNews-vectors.
BERT pakage:  You can download package from: https://github.com/google-research/bert. I used BERT-Large Cased.

## How to run
#### Get tweets:
Replace the Twitter API credentials in "TwitterDumper.py", and run "getTweets.py" to get twiiter by specific key words.
#### Topic detection:
Run analyze.py in "Topic detection" to classify tweets by topic: the input data is "all_tweets" in folder "data". It will create an output .csv for each topic.

This part refer to: https://github.com/AaronJi/TrumpTwitterAnalysis

#### Crowdsourcing:
The interface's code of topic " Hold breathe" is in the folder "AMT".
#### Misinformation detection:
After crowdsourcing platform, we can get the labeled results on each topic. Run "contact.py" to contact all topic tweets to form one csv. The output is "Data_AMT_raw.csv".
Extract the labeled results through "extract.py" to get "train_AMT.csv". Use "clean.py" to clear USERNAME, Tag, URL in the tweets. Use "sample.py" to divide the file into a training set and a data set. 
The result of the above steps are "train.csv" and "test.csv" in the "data" folder
Call the support vector machine by running "SVM.py", the input should be "train_raw.csv" in "Data" folder.
Call the BERT model by running "run_classifier.py" in the "BERT" folder, the input should be "train.csv" and "test.csv". And "acc.py" in the "BERT" folder is the correctness verification file for the BERT model. BERT model refer to: https://github.com/google-research/bert


