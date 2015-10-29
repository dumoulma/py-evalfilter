#!/usr/bin/env python

import logging
import os
import time
import datetime

import click
import numpy as np
import scipy as sp

from sklearn.feature_extraction import DictVectorizer

from sklearn.datasets import dump_svmlight_file

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline

from datasets.fuman_base import load_fuman_rants
from datasets.output import save_dataset_metadata2, save_features_json
from datasets.fuman_features import vectorize_text, vectorise_text_fit
from util.mecab import tokenize_pos
from datasets.fuman_features import RantStats, get_header
from datasets.fuman_base import fuman_gvb_target

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ALLSCORED = 'all-scored-rants.csv'
VECTORIZERS = {'tfidf': TfidfVectorizer, 'count': CountVectorizer}


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--split_size', default=10000)
@click.option('--max_splits', default=2)
@click.option('--pos_max_features', default=5000)
@click.option('--pos_bad_only', is_flag=True)
@click.option('--pos_min_df', default=100)
@click.option('--pos_ngram', default=3)
@click.option('--pos_vec', type=click.Choice(['tfidf', 'count']), default='count')
@click.option('--sparse', is_flag=True)
@click.option('--simple_headers', is_flag=True)
def main(source, output, split_size, max_splits, pos_max_features, pos_min_df, pos_bad_only, pos_vec, pos_ngram, sparse,
         simple_headers):
    """
    Generates a good vs bad training dataset from Fuman user posts. (Binary Classification)

    Concatenates simple features from the database, hand crafted features based on various character and word counts,
    and Tf-Idf weighted bag of words based on the text as well as the part-of-speech tags of Fuman user posts.

    :param source: directory or file of the input files. (If dir, file will be all-scored-rants.csv)
    :param output: the output directory
    :param split_size: the size (in instances) of each n_splits of the data
    :param max_splits: the number of splits to generate
    :param pos_max_features: parameter for tf-idf vectorizer (default 50000)
    :param pos_min_df: parameter for tf-idf vectorizer (default 100)
    :param pos_bad_only: learn vocabulary for POS from bad rants only (flag, default is all dataset)
    :param pos_vec: [tfidf, count] use corresponding term weighting
    :param pos_ngram: Learn vocabulary with ngrams in range (1,pos_ngram) (default is 3)
    """
    if not os.path.isdir(output):
        raise ValueError("Output must be a directory")

    if os.path.isfile(source):
        source_filepath = source
    else:
        source_filepath = os.path.join(source, ALLSCORED)
    logging.info("Source dump: {}".format(source_filepath))
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    logging.info("Timestamp: {}".format(timestamp))
    output_path = os.path.join(output, "gvb-" + timestamp)
    output_filename = os.path.join(output_path, "{}-{}.csv".format("goodvsbad", timestamp))
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

    logging.info("Processing pipeline...")
    fuman_data = load_fuman_rants(source_filepath, fuman_gvb_target)
    X = pipeline.fit_transform(fuman_data.data)
    save_features_json(pos_dict_filename, vectorizer.get_feature_names())
    headers = get_header()
    if not pos_max_features:
        headers += ',' + ','.join(vectorizer.get_feature_names())
    headers += ',target'

    if sparse:
        y = np.asarray(fuman_data.target, dtype=np.float64).reshape((X.shape[0],))
        with open(output_filename, mode='wb') as f:
            dump_svmlight_file(X, y, f)
    else:
        y = np.asarray(fuman_data.target, dtype=np.float64).reshape((X.shape[0], 1))
        all_data = sp.sparse.hstack([X, y]).todense()
        np.savetxt(output_filename, all_data, delimiter=',', header=headers)
        save_dataset_metadata2(sparse, output_path, "goodvsbad", pos_max_features, pos_min_df, pos_ngram, pos_vec_func,
                               vectorizer, source_filepath, timestamp, tokenize_pos)
    logging.info("Work complete!")


def get_pos_header(features, simple_headers):
    headers = features
    if simple_headers:
        headers = ["pos_" + str(i) for i in range(len(features))]
    return ','.join(headers)


if __name__ == "__main__":
    main()
