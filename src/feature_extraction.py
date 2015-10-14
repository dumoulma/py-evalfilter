#!/usr/bin/env python
import logging

import click
from datasets.fuman_raw import load_fuman_csv, load_rants
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import sklearn.preprocessing as pp
import numpy as np
from mecab import tokenize_pos, tokenize_rant

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


@click.command()
@click.argument('filename', type=click.Path(), nargs=1)
def main(filename):
    wdvec = TfidfVectorizer(tokenizer=tokenize_rant, strip_accents='unicode', min_df=100, max_features=5000)
    rants_vects = wdvec.fit_transform(load_rants(filepath=filename))
    print(rants_vects.shape)
    posvec = TfidfVectorizer(tokenizer=tokenize_pos, ngram_range=(1, 3), strip_accents='unicode', min_df=10,
                             max_features=5000)
    pos_vects = posvec.fit_transform(load_rants(filepath=filename))
    wd_pos_vecs = sp.hstack((rants_vects, pos_vects), format='csr')
    print(pos_vects.shape)
    X = list(load_fuman_csv(filename, target_var_func=set_goodvsbad_label))
    print("{} negatives found".format(abs(sum(filter(lambda x: x is 1, [x[-1] for x in X])))))
    logging.info("Got {} rows".format(len(X)))


# def encode_categoricals(X):
#     age_enc = pp.LabelEncoder()
#     encoded_age = age_enc.fit_transform([x[6] for x in X])
#     state_enc = pp.LabelEncoder()
#     encoded_state = state_enc.fit_transform([x[7] for x in X])
#     job_enc = pp.LabelEncoder()
#     encoded_job = job_enc.fit_transform([x[9] for x in X])
#     for x, ea, es, ej in zip(X, encoded_age, encoded_state, encoded_job):
#         x[6] = ea
#         x[7] = es
#         x[9] = ej


def set_goodvsbad_label(status, _):
    if status is 100:
        return -1
    elif 200 <= status < 300:
        return 1
    else:
        raise ValueError("Unexpected value for status")


def set_price(_, price):
    return price


if __name__ == "__main__":
    main()
