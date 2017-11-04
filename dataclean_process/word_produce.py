import re
from tqdm import tqdm
import hashlib
import pandas as pd
from collections import defaultdict
import numpy as np


md5 = lambda s: hashlib.md5(s).hexdigest()
def works():
    texts_set = set()
    df = pd.read_excel('report.xls')
    for a in tqdm(df['内容']):
        pass
        if md5(a.encode('utf-8')) in texts_set:
            continue
        else:
            texts_set.add(md5(a.encode('utf-8')))
            # print(texts_set)
            for t in re.split(u'[^\u4e00-\u9fa5]+', a):
                if t:
                    # print(t)
                    yield t
    print('最终计算了%s篇文章' % len(texts_set))

# def count():
n = 3
min_count = 150
ngrams = defaultdict(int)

for t in works():
    for i in range(len(t)):
        for j in range(1, n+1):
            if i+j <= len(t):
                ngrams[t[i:i+j]] += 1

ngrams = {i:j for i,j in ngrams.items() if j >= min_count}
total = 1.*sum([j for i,j in ngrams.items() if len(i) == 1])

min_proba = {2:5, 3:25, 4:125}

def is_keep(s, min_proba):
    if len(s) >= 2:
        score = min([total*ngrams[s]/(ngrams[s[:i+1]]*ngrams[s[i+1:]]) for i in range(len(s)-1)])
        if score > min_proba[len(s)]:
            return True
    else:
        return False

ngrams_ = set(i for i,j in ngrams.items() if is_keep(i, min_proba))

def cut(s):
    r = np.array([0]*(len(s)-1))
    for i in range(len(s)-1):
        for j in range(2, n+1):
            if s[i:i+j] in ngrams_:
                r[i:i+j-1] += 1
    w = [s[0]]
    for i in range(1, len(s)):
        if r[i-1] > 0:
            w[-1] += s[i]
        else:
            w.append(s[i])
    return w

words = defaultdict(int)
for t in works():
    for i in cut(t):
        words[i] += 1

words = {i:j for i,j in words.items() if j >= min_count}

def is_real(s):
    if len(s) >= 3:
        for i in range(3, n+1):
            for j in range(len(s)-i+1):
                if s[j:j+i] not in ngrams_:
                    return False
        return True
    else:
        return True

w = {i:j for i,j in words.items() if is_real(i)}


def to_text():
    # for word in w.items():
    #     print(word)
    with open('merge.txt', 'w') as f:
        for i, j in w.items():
            # if j < 100:
                f.write(i+'\n')


if __name__ == '__main__':
    to_text()
