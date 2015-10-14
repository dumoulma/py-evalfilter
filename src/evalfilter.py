__author__ = 'dumoulma'

import logging
import unicodedata

import MeCab

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

X = []
with open("data/20150930/good-vs-bad-100.csv") as f:
    f.readline(1)
    for line in f:
        X.append(line.split(","))

for x in X:
    print(x)

# init mecab
# load csv
# generate and transform the features
# computed features from text
# generate features for categorical variables
# generate TF-IDF word features
# generate TF-POS features