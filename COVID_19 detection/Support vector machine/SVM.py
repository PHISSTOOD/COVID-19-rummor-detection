import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv("C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Bert_data\\train_raw.csv")

import nltk
nltk.download('stopwords')

corpus = []
#iterate all tweets in train data set
for i in range(0, 305):
    review = dataset.iloc[i][0]
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer()
vect =CountVectorizer(min_df=0.,max_df=1.0)
X = cv.fit_transform(corpus)
G = vect.fit_transform(corpus)
y = dataset.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X, y)
y_pred = classifier.predict(X_test)

def show_cm(pre_label, true_label):
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
print(y_pred)
print(y_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
show_cm(y_pred, y_test)

