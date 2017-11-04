# coding:utf-8
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import time

class TfIdf(object):
    def __init__(self):
        pass

    def tfidf(self):
        df = pd.read_csv('~/Desktop/NLP/data/last.csv', encoding='utf-8', na_values='NULL')
        df = df.dropna()
        corpus = [doc.replace(',', ' ') for doc in df['content']]
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        word = vectorizer.get_feature_names()
        weight = tfidf.toarray()
        for i in range(len(weight)):
            print (u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
            for j in range(len(word)):
                print (word[j], weight[i][j])
            time.sleep(5)

if __name__ == '__main__':
    re = TfIdf()
    re.tfidf()