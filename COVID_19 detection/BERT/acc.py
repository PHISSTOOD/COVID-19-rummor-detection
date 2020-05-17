#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: lqhou
@file: calculate_acc.py
@time: 2020/02/15
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def show_cm(pre_label, true_label):
    """绘制混淆矩阵"""
    classes = ['true', 'fake']
    confusion = confusion_matrix(pre_label, true_label)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Predict')
    plt.ylabel('True')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])

    plt.show()


t2t_pre_data = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\output\\test_results.tsv"
t2t_true_data = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\test.csv"

pre_data = pd.read_csv(t2t_pre_data, sep='\t', header=None)
predict = np.argmax(np.array(pre_data), axis=1)

true = np.array(pd.read_csv(t2t_true_data, sep=',', header=None, usecols=[1]))
true = true.reshape(-1)

print(predict)
print(true)

print(classification_report(true, predict))

print(accuracy_score(true, predict))

show_cm(predict, true)
