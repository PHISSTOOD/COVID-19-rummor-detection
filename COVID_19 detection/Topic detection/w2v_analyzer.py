from gensim.models import Phrases, Word2Vec
import multiprocessing
import cvxpy as cvx
import numpy as np
from basic_analysis import subsList

class w2v_analyzer(object):
    # methodType: 0: CBOW; 1: skip-gram
    def __init__(self, source, methodType, nFeature, niter, ignoredWordSet):
        self.nFeature = nFeature
        self.niter = niter
        self.ignoredWordSet = ignoredWordSet
        self.vocab = set()
        self.status = 0  # 0: not initiated; 1: initiated, can be trained; -1: initiated, can not be traned
        min_count = 1
        if source == 'GoogleNews':
            from gensim.models.keyedvectors import KeyedVectors
            import os
            current_folder = os.path.dirname(os.path.realpath(__file__))
            self.nFeature = 300
            self.model = Word2Vec(iter=1, sg=methodType, size=self.nFeature, min_count=min_count, window=5, workers=multiprocessing.cpu_count())
            self.model.wv = KeyedVectors.load_word2vec_format("C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\W2V\\GoogleNews-vectors-negative300.bin", binary=True)

            self.status = -1
        else:
            # initilaze with given texts
            bigram = False
            if bigram:
                bigram_transformer = Phrases(source)
                self.model = Word2Vec(bigram_transformer[source], sg=methodType, iter=self.niter, size=self.nFeature, min_count=min_count, window=5, workers=multiprocessing.cpu_count())
            else:
                self.model = Word2Vec(source, sg=methodType, iter=self.niter, size=self.nFeature, min_count=min_count, window=5, workers=multiprocessing.cpu_count())
        return

    def fit(self, texts):
        if self.status == 0:
            self.model.build_vocab(texts)
        if self.status >= 0:
            print("Train docs " + str(len(texts)))
            self.model.train(texts, total_examples=self.model.corpus_count, epochs=self.niter)
        return

    def transform(self, texts):
        X = np.zeros((len(texts), self.nFeature))
        for i, text in enumerate(texts):
            sv = self.getSenVec(text)
            X[i, :] = sv
        return X

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def getSimilarity(self, word1, word2):
        return self.wv.similarity(word1, word2)

    def getSimilarWords(self, word, n=5):
        similarResults = self.model.wv.most_similar(positive=word, topn=n)
        simWords = list()
        simNums = list()
        for sR in similarResults:
            simNums.append(sR[1])
        return simWords, simNums

    def getSimilarWordsFromList(self, posWordList, negWordList, n):

        similarResults = self.model.wv.most_similar(positive=posWordList, negative=negWordList, topn=n)

        simWords = list()
        simNums = list()
        for sR in similarResults:
            simWords.append(sR[0])
            simNums.append(sR[1])

        return simWords, simNums

    def getUnmathedWord(self, wordList):
        return self.model.wv.doesnt_match(wordList)

    def wvs2sv(self, wvs, d):
        #sv = np.mean(wvs, axis=0)
        sv = np.zeros(self.nFeature)
        for i in range(d.shape[0]):
            sv += d[i]*wvs[i]
        return sv

    # given a list of words, return a matrix with each row as a word's vector
    def getSenMat(self, words):
        wvs = None
        words_queue = list()
        d = list()
        for word in words:
            if word not in self.ignoredWordSet:
                if word in subsList:
                    word = subsList[word]
                if word in words_queue:
                    i = words_queue.index(word)
                    d[i] += 1
                else:
                    try:
                        wv = self.model.wv[word]
                        if wvs is None:
                            wvs = wv.copy().reshape(1, -1)
                        else:
                            wvs = np.vstack((wvs, wv))

                        words_queue.append(word)
                        d.append(1)
                    except KeyError:
                        continue

        d = np.array(d, dtype=float)
        d /= float(np.sum(d))
        if wvs is not None:
            assert len(words_queue) == wvs.shape[0] == d.shape[0]

        return wvs, d

    def getSenVec(self, words):
        wvs, d = self.getSenMat(words)
        return self.wvs2sv(wvs, d)

    def WMD(wvs1, wvs2, d1, d2):
        n1 = d1.shape[0]
        n2 = d2.shape[0]

        assert wvs1.shape[0] == n1
        assert wvs2.shape[0] == n2
        assert wvs1.shape[1] == wvs2.shape[2]

        wwdist = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                wwdist[i, j] = np.linalg.norm(wvs1[i] - wvs2[j], ord=2)

        T = cvx.Variable(n1, n2)

        obj = cvx.sum_entries(cvx.mul_elemwise(wwdist, T))

        cons = [cvx.sum_entries(T, axis=1) == d1,
                cvx.sum_entries(T, axis=0) >= d2,
                0 <= T]

        prob = cvx.Problem(cvx.Maximize(obj), cons)

        # Solve
        prob.solve(solver=cvx.ECOS)  # , mi_max_iters=100
        Topt = T.value
        senDiff = prob.value

        return senDiff, Topt, prob.status

    def getSenDiff(self, words1, words2, senDiffType):

        wvs1, d1 = self.getSenMat(words1)
        wvs2, d2 = self.getSenMat(words2)

        if wvs1 is None or wvs2 is None:
            return -1

        if senDiffType >= 2:
            senDiff, Topt, LPstatus = self.WMD(wvs1, wvs2, d1, d2)
        else:
            sv1 = self.wvs2sv(wvs1, d1)
            sv2 = self.wvs2sv(wvs2, d2)
            senDiff0 = np.linalg.norm(sv1-sv2, 2)
            if senDiffType == 0:
                # WCD
                senDiff = senDiff0
            elif senDiffType == 1:
                # RWMD
                senDiff = 0
                for i in range(wvs1.shape[0]):
                    cij_min = np.inf
                    for j in range(wvs2.shape[0]):
                        cij = np.linalg.norm(wvs1[i]-wvs2[j], 2)
                        if cij < cij_min:
                            cij_min = cij
                    senDiff += d1[i]*cij_min
                #senDiff = max(senDiff, senDiff0)
        return senDiff