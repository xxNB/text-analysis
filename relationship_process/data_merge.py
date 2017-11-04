# 将从LDA筛选出来的主题词进行去重， 战狼csv转换成txt

import pandas as pd

def txt_process():
    with open('topic_word.txt', 'r') as f:
        with open('new_topic_word.txt', 'w') as e:
            word_list = [i for i in f.readlines()]
            world_set = set(word_list)
            for n in world_set:
                e.write(n)

def comment_txt():
    df = pd.read_csv('~/Desktop/NLP/data/merge.csv', encoding='utf-8')
    with open('merge.txt', 'w') as f:
        for text in df['content']:
            f.write(text)

if __name__ == '__main__':
    comment_txt()

