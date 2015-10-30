#!/usr/bin/env python

import logging
import os
import time
import datetime

import click
import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn.cross_validation import StratifiedKFold

from sklearn.datasets import dump_svmlight_file

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline

from datasets.fuman_base import load_fuman_gvb
from datasets.output import save_dataset_metadata2, save_features_json
from datasets.fuman_features import vectorize_text, vectorise_text_fit
from util.mecab import tokenize_pos
from datasets.fuman_features import RantStats, get_header
from util.file import get_size

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

GOOD_FILENAME = "good-rants.csv"
BAD_FILENAME = "bad-rants.csv"
VECTORIZERS = {'tfidf': TfidfVectorizer, 'count': CountVectorizer}


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--n_folds', default=5)
@click.option('--pos_max_features', default=3000)
@click.option('--pos_bad_only', is_flag=True)
@click.option('--pos_min_df', default=25)
@click.option('--pos_ngram', default=2)
@click.option('--pos_vec', type=click.Choice(['tfidf', 'count']), default='count')
@click.option('--sparse', is_flag=True)
@click.option('--simple_header', is_flag=True)
def main(source, output, n_folds, pos_max_features, pos_min_df, pos_bad_only, pos_vec, pos_ngram, sparse,
         simple_header):
    """
    Generates a good vs bad training dataset from Fuman user posts. (Binary Classification)

    Concatenates simple features from the database, hand crafted features based on various character and word counts,
    and Tf-Idf weighted bag of words based on the text as well as the part-of-speech tags of Fuman user posts.

    :param source: directory or file of the input files. (If dir, file will be all-scored-rants.csv)
    :param output: the output directory
    :param n_folds: the number of splits to generate (using StratifiedKFold)
    :param pos_max_features: parameter for tf-idf vectorizer (default 50000)
    :param pos_min_df: parameter for tf-idf vectorizer (default 100)
    :param pos_bad_only: learn vocabulary for POS from bad rants only (flag, default is all dataset)
    :param pos_vec: [tfidf, count] use corresponding term weighting
    :param pos_ngram: Learn vocabulary with ngrams in range (1,pos_ngram) (default is 3)
    """
    if not os.path.isdir(output):
        raise ValueError("Output must be a directory")

    if os.path.isfile(source):
        raise ValueError("Source must be a directory")
    logging.info("Source dump: {}".format(source))
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    logging.info("Timestamp: {}".format(timestamp))
    output_path = os.path.join(output, "gvb-" + timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pos_vec_func = vectorize_text
    if pos_bad_only:
        pos_vec_func = vectorise_text_fit
    pos_vectorizer_func = VECTORIZERS[pos_vec]
    pos_dict_filename = os.path.join(output_path, "pos-vocabulary-" + timestamp + ".json")
    vectorizer = pos_vectorizer_func(tokenizer=tokenize_pos, ngram_range=(1, pos_ngram), strip_accents='unicode',
                                     min_df=pos_min_df, max_features=pos_max_features)
    pipeline = Pipeline([('stats', RantStats()),  # returns a list of dicts
                         ('vect', DictVectorizer())])  # list of dicts -> feature matrix
    if pos_max_features:
        pipeline = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[
                    ('body_stats', pipeline),
                    ("pos_vec", vectorizer),
                ],
                n_jobs=1))
        ])

    fuman_data = load_fuman_gvb(source, BAD_FILENAME, GOOD_FILENAME)
    logging.info("Processing pipeline...")
    X = pipeline.fit_transform(fuman_data.data)
    n_samples = X.shape[0]
    y = np.asarray(fuman_data.target, dtype=np.int8).reshape((n_samples,))

    save_features_json(pos_dict_filename, vectorizer.get_feature_names())
    logging.info("Saving {} folds to disk...".format(n_folds))
    skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
    for i, (_, test_index) in enumerate(skf, 1):
        dump_csv(output_path, X[test_index], y[test_index], vectorizer.get_feature_names(), i, timestamp,
                 simple_header, sparse)
    save_dataset_metadata2(sparse, output_path, "goodvsbad", pos_max_features, pos_min_df, pos_ngram, pos_vec_func,
                           vectorizer, source, timestamp, tokenize_pos)
    logging.info("Work complete!")


def dump_csv(output_path, X, y, pos_features, nth_fold, timestamp, simple_header, sparse):
    output_filename = os.path.join(output_path, "{}-{}-{}.csv".format("goodvsbad", timestamp, nth_fold))
    if sparse:
        with open(output_filename, mode='wb') as f:
            dump_svmlight_file(X, y, f)
        return
    n_samples = X.shape[0]
    header = get_header()
    n_pos_features = len(pos_features)
    if n_pos_features:
        if simple_header:
            header += ',' + ','.join("pos_" + str(i) for i in range(n_pos_features))
        else:
            header += ',' + ','.join(pos_features)
    header += ',target'
    y = y.reshape((n_samples, 1))
    import scipy as sp
    all_data = sp.sparse.hstack([X, y]).todense()
    sp.savetxt(output_filename, all_data, fmt='%.3f', delimiter=',', header="#" * len(header))
    import fileinput
    with fileinput.FileInput(output_filename, inplace=1) as fi:
        fi.readline()
        print(header)
        for line in fi:
            print(line, end='')
    # regex = re.compile(r'([0-9]).0,')  # replace "X.0" -> "X" to make csv smaller
    # with open(output_filename, mode='w') as f:
    #     f.write(headers+'\n')
    #     for i in range(n_samples):
    #         row = ','.join(str(X[i, j]) for j in range(n_features))
    #         row = re.sub(regex, r'\1,', row)
    #         f.write('{},{}\n'.format(row, y[i][0]))
    logging.info("Wrote fold {} to {} ({} instances {} MB)".format(nth_fold, output_filename, n_samples,
                                                                   get_size(output_filename)))


def get_pos_header(features, simple_headers):
    headers = features
    if simple_headers:
        headers = ["pos_" + str(i) for i in range(len(features))]
    return ','.join(headers)


if __name__ == "__main__":
    main()
