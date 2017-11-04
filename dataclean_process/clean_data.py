# coding: utf-8
import jieba
import pandas as pd
import re
import codecs
from dataclean_process.word_produce import to_text

syn = dict()
with codecs.open('word_synonym.txt', 'r') as f:
    for line in f.readlines():
        for word in line.strip('\n').split(' ')[1:]:
            print(word)
            syn[word] = line.strip('\n').split(' ')[0]
print(syn)

class CleanData(object):
    def __init__(self, dir):
        self.dir = dir
        self.main()
        pass

    def produce_neword(self):
        to_text()

    def made(self):
        df = pd.read_excel(self.dir)
        jieba.load_userdict('merge.txt')
        df1 = pd.DataFrame()
        df1['content'] = df['内容'].apply(self.clean)
        df1.to_csv('jiebacut.csv')
        print (df1.columns)


    @staticmethod
    def clean(text):
        final = []
        print(text)
        r1 = u'[^\u4e00-\u9fa5]+'
        new_text = re.sub(r1, '', text.strip())
        seg_list = jieba.lcut(new_text, cut_all=False)
        print(seg_list)
        stopwords = [line.rstrip() for line in open('stopword.txt')]
        for seg in seg_list:
            # print(seg)
            if seg not in stopwords:
                if seg in syn.keys():
                    new_seg = syn[seg]
                    final.append(new_seg)
                final.append(seg)
        # print(final)
        return final


    # @staticmethod
    # def synonym_clean(text):
    #     final = []
    #     for word in text:
    #         if word in syn.keys():
    #             print('be replace', word)
    #             new_word = syn[word]
    #             final.append(new_word)
    #         final.append(word)
    #     new_text = ','.join(final)
    #     print(new_text)
    #     return new_text

    def main(self):
        self.produce_neword()
        self.made()

if __name__ == '__main__':
    res = CleanData('report.xls')
