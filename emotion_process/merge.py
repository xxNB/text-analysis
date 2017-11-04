# coding:utf-8

"""
用于训练word2vec词向量model，语料来自搜狗实验室全网无标注新闻素材
"""
import jieba
import re
import codecs
import os


def readLines(filename):
    # read txt or csv file
    fp = open(filename, 'r')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        line = line.decode("utf-8")
        lines.append(line)
    fp.close()
    return lines

def parseSent(sentence):
    seg_list = jieba.cut(sentence)
    output = ' '.join(list(seg_list)) 
    return output

    
    pattern = "<content>(.*?)</content>"
    csvfile = codecs.open("corpus.csv", 'w', 'utf-8')
    fileDir = os.listdir("/Users/zhangxin/Downloads/SogouCA/")
    print(fileDir)
    for file in fileDir:
        print(u'开始处理', file)
        with open("/Users/zhangxin/Downloads/SogouCA/%s" % file, "r") as txtfile:
            for ix, line in enumerate(txtfile):
                m = re.match(pattern, line)
                if m:
                    print(ix)
                    segSent = parseSent(m.group(1))
                    csvfile.write("%s" % segSent)

    from gensim.models import word2vec
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("corpus.csv")  # 加载语料
    model = word2vec.Word2Vec(sentences, size=400)  # 训练skip-gram模型

    # 保存模型，以便重用
    model.save("corpus.model")
if __name__ == '__main__':
    # from __future__ import absolute_import
    # from gensim.models import word2vec
    # import pattern
    import gensim
    # model = word2vec.Word2Vec.load("corpus.model")
    # model = word2vec.Word2Vec.load("corpus.model")
    # 以一种C语言可以解析的形式存储词向量
    # model.wv.save_word2vec_format("corpus.model.bin", binary=True)
    # 对应的加载方式
    # model = gensim.models.KeyedVectors.load_word2vec_format("corpus.model.bin", binary=True)
    # print(model.most_similar(u'安徽工程大学'))
    # print ("man woman kitchen child".split())