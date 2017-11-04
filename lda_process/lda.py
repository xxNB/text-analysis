# coding:utf-8
from gensim import corpora
import gensim
import pandas as pd

class Lda(object):

    def __init__(self):
        pass

    def before_lda(self):
        texts = []
        df = pd.read_csv('~/Desktop/NLP/data/last.csv', encoding='utf-8', na_values='NULL')
        df = df.dropna()
        for docs in df['content']:
            texts.append([word for word in docs.split(',')])
        return texts

    def made_corpus(self, texts):
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        return dictionary, corpus

    def lda(self):
        texts = self.before_lda()
        dictionary = self.made_corpus(texts)[0]
        corpus = self.made_corpus(texts)[1]
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
        return lda

if __name__ == '__main__':
    rel = Lda()
    lda_model = rel.lda()
    # 打印其中一个主题(比如说第11个主题，选出前5关键词)
    print (lda_model.print_topic(8, topn=5))
    # 打印所有主题
    res = lda_model.print_topics(num_topics=20, num_words=10)
    word_list = []
    for i in res:
        text = i[1].encode('utf-8')
        print(text)
        # 取出主题词用于网生成络语义图
    #     pattern = re.compile(u"[\u4e00-\u9fa5]+")
    #     gen = pattern.findall(text.decode('utf-8'))
    #     for word in gen:
    #         word_list.append(word)
    # print set(word_list)

