# coding:utf-8
import os
import numpy as np
import gensim
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC

class Emotion(object):
    def __init__(self):
        """
        生成模型
        """
        # 生成训练词向量以及将训练材料转换成numpy数组
        # self.get_train_vecs()
        # 训练情感分析的model
        # self.svm_train()
        pass

    def load_file_and_preprocessing(self):
        neg = pd.read_excel(os.path.join(os.path.dirname(__file__), 'data', 'neg.xls'), header=None, index=None)
        pos = pd.read_excel(os.path.join(os.path.dirname(__file__), 'data', 'pos.xls'), header=None, index=None)

        cw = lambda x: list(jieba.cut(x.encode('utf-8')))
        pos['words'] = pos[0].apply(cw)
        neg['words'] = neg[0].apply(cw)
        print(len(pos['words']))
        print(len(neg['words']))
        # use 1 for positive sentiment, 0 for negative
        y_train = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
        x_train = np.concatenate((pos['words'], neg['words']))
        np.save('svm_data/y_train.npy', y_train)
        np.save('svm_data/x_train.npy', x_train)
        yield x_train

    def build_sentence_vector(self, text, size,imdb_w2v):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in text:
            try:
                vec += imdb_w2v[word].reshape((1, size))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    def get_train_vecs(self, ):
        """将训练词语转化成word2vec模型"""
        for x_train in self.load_file_and_preprocessing():
            n_dim = 400
            model = gensim.models.KeyedVectors.load_word2vec_format("corpus.model.bin", binary=True, unicode_errors='ignore')
            train_vecs = np.concatenate([self.build_sentence_vector(z, n_dim, model) for z in x_train])
            np.save('svm_data/train_vecs.npy', train_vecs)
            print(train_vecs.shape)


    def get_data(self):
        train_vecs = np.load('svm_data/train_vecs.npy')
        y_train = np.load('svm_data/y_train.npy')
        x_train = np.load('svm_data/x_train.npy')
        return train_vecs, y_train, x_train

    def svm_train(self):
        train_vecs = self.get_data()[0]
        y_train = self.get_data()[1]
        clf=SVC(kernel='rbf',verbose=True)
        print (train_vecs, train_vecs.shape)
        print (y_train, y_train.shape)
        clf.fit(train_vecs, y_train)
        joblib.dump(clf, 'svm_data/svm_model/model.pkl')


    def get_predict_vecs(self, words):
        n_dim = 400
        model = gensim.models.KeyedVectors.load_word2vec_format("corpus.model.bin", binary=True, unicode_errors='ignore')
        train_vecs = self.build_sentence_vector(words, n_dim, model)
        return train_vecs

    def svm_predict(self, string):
        words = jieba.lcut(string)
        words_vecs = self.get_predict_vecs(words)
        clf = joblib.load('svm_data/svm_model/model.pkl')

        result = clf.predict(words_vecs)
        if int(result[0]) == 1:
            print('positive')
            return 'positive'
        else:
            print('negative')
            return 'negative'

    def doc_predict(self):
        """
        测试豆瓣影评的情感分析，占比情况
        :return: 
        """
        df = pd.read_csv('merge.csv', na_values='NULL').dropna()
        df['sentiment'] = df['content'].apply(self.svm_predict)
        df.to_excel('sentiment.xls', encoding='utf-8')
        df1 = pd.read_excel('sentiment.xls')
        result = df1['sentiment'].tolist()
        pos_times = result.count('positive')
        neg_times = result.count('negative')
        print('pos_times:', pos_times)
        print('neg_times:', neg_times)

if __name__ == '__main__':
    res = Emotion()
    res.svm_predict('你就是个傻子')
    # 对冈波仁齐影评的每篇文档做情感极性分析
    # res.doc_predict()
    pass



