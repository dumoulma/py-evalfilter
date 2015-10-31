#!/usr/bin/env python

import logging
import os
import time
import datetime
import fileinput
import warnings

import click
import numpy as np
from scipy import savetxt
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline

from datasets.fuman_base import load_fuman_price
from datasets.output import save_dataset_metadata, save_features_json
from util.mecab import tokenize_pos, tokenize_rant
from datasets.fuman_features import FieldSelector, RantStats, UserProfileStats, get_header_userprofile
from util.file import get_size

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

GOOD_FILENAME = "good-rants.csv"
BAD_FILENAME = "bad-rants.csv"
PRICE_FILENAME = "rants-price.csv"
VECTORIZERS = {'tfidf': TfidfVectorizer, 'count': CountVectorizer}


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--n_folds', default=5)
@click.option('--n_folds_max', default=2)
@click.option('--pos_max_features', default=3000)
@click.option('--pos_min_df', default=25)
@click.option('--rant_max_features', default=3000)
@click.option('--rant_min_df', default=25)
@click.option('--pos_ngram', default=2)
@click.option('--pos_vec', type=click.Choice(['tfidf', 'count']), default='count')
@click.option('--sparse', is_flag=True)
@click.option('--feature_name_header', is_flag=True)
def main(source, output, n_folds, n_folds_max, rant_max_features, rant_min_df, pos_max_features, pos_min_df,
         pos_vec, pos_ngram, sparse, feature_name_header):
    """
    Generates a good vs bad training dataset from Fuman user posts. (Binary Classification)

    Concatenates simple features from the database, hand crafted features based on various character and word counts,
    and Tf-Idf weighted bag of words based on the text as well as the part-of-speech tags of Fuman user posts.

    :param source: directory or file of the input files. (If dir, file will be all-scored-rants.csv)
    :param output: the output directory
    :param n_folds: the number of splits to generate (using StratifiedKFold)
    :param n_folds_max: max number of folds to output
    :param pos_max_features: parameter for tf-idf vectorizer (default 50000)
    :param pos_min_df: parameter for tf-idf vectorizer (default 100)
    :param pos_vec: [tfidf, count] use corresponding term weighting
    :param pos_ngram: Learn vocabulary with ngrams in range (1,pos_ngram) (default is 3)
    :param sparse: output in svmlight sparse format
    :param feature_name_header: output headers as the feature names
    """
    if not os.path.isdir(output):
        raise ValueError("Output must be a directory")

    if os.path.isfile(source):
        source_dir, source_filename = os.path.split(source)
    else:
        source_dir = source
        source_filename = PRICE_FILENAME
    logging.info("Source dump: {}/{}".format(source_dir, source_filename))

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    output_path = os.path.join(output, "price-{}".format(timestamp))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.info("Timestamp: {}".format(timestamp))

    pos_vectorizer_func = VECTORIZERS[pos_vec]
    rant_dict_filename = os.path.join(output_path, "rant-features-" + timestamp + ".json")
    pos_dict_filename = os.path.join(output_path, "pos-features-" + timestamp + ".json")
    pos_vectorizer = pos_vectorizer_func(tokenizer=tokenize_pos, ngram_range=(1, pos_ngram), strip_accents='unicode',
                                         min_df=pos_min_df, max_features=pos_max_features)
    rant_vectorizer = TfidfVectorizer(tokenizer=tokenize_rant, strip_accents='unicode', min_df=rant_min_df,
                                      max_features=rant_max_features)
    pipeline = get_pipeline(pos_max_features, pos_vectorizer, rant_max_features, rant_vectorizer)

    fuman_data = load_fuman_price(source_dir, filename=source_filename)

    logging.info("Processing pipeline...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="sklearn")
        X = pipeline.fit_transform(fuman_data.data)
        n_samples = X.shape[0]
        y = np.asarray(fuman_data.target, dtype=np.int8).reshape((n_samples,))

        save_features_json(pos_dict_filename, pos_vectorizer.get_feature_names())
        save_features_json(rant_dict_filename, rant_vectorizer.get_feature_names())

        logging.info("Saving {} of {} folds to disk...".format(n_folds_max, n_folds))
        skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
        for i, (_, test_index) in enumerate(skf, 1):
            dump_csv(output_path, X[test_index], y[test_index], i, pos_vectorizer.get_feature_names(),
                     rant_vectorizer.get_feature_names(), timestamp, feature_name_header, sparse)
            if i == n_folds_max:
                break
        save_dataset_metadata(sparse, output_path, "price",
                              pos_vectorizer=pos_vectorizer, source_filepath=source, timestamp=timestamp,
                              word_vectorizer=rant_vectorizer, tokenize_rant=tokenize_rant, tokenize_pos=tokenize_pos)
    logging.info("Work complete!")


def get_pipeline(pos_max_features, pos_vectorizer, rant_max_features, rant_vectorizer):
    transformer_list = [
        ('rant_stats', Pipeline([
            ('selector', FieldSelector(key='rant')),
            ('stats', RantStats()),  # returns a list of dicts
            ('vect', DictVectorizer()),  # list of dicts -> feature matrix
        ])),
        ('userprofile_stats', Pipeline([
            ('selector', FieldSelector(key='userprofile')),
            ('stats', UserProfileStats()),  # returns a list of dicts
            ('vect', DictVectorizer()),  # list of dicts -> feature matrix
        ])),
    ]

    if pos_max_features:
        transformer_list.append(('pos_bow', Pipeline([
            ('selector', FieldSelector(key='rant')),
            ('vectorize', pos_vectorizer),
        ])))
    if rant_max_features:
        transformer_list.append(('rant_bow', Pipeline([
            ('selector', FieldSelector(key='rant')),
            ('vectorize', rant_vectorizer),
        ])))
    return Pipeline([
        ('union', FeatureUnion(transformer_list=transformer_list))
    ])


def dump_csv(output_path, X, y, nth_fold, pos_features, rant_features, timestamp, feature_name_header, sparse):
    output_filename = os.path.join(output_path, "{}-{}-{}.csv".format("price", timestamp, nth_fold))
    if sparse:
        with open(output_filename, mode='wb') as f:
            dump_svmlight_file(X, y, f)
        return
    n_samples = X.shape[0]
    header = get_header_userprofile()
    n_pos_features = len(pos_features)
    if n_pos_features:
        if feature_name_header:
            header += ',' + ','.join(pos_features)
        else:
            header += ',' + ','.join("pos_" + str(i) for i in range(n_pos_features))
    n_rant_features = len(rant_features)
    if n_rant_features:
        if feature_name_header:
            header += ',' + ','.join(rant_features)
        else:
            header += ',' + ','.join("word_" + str(i) for i in range(n_rant_features))
    header += ',target'
    y = y.reshape((n_samples, 1))
    all_data = hstack([X, y]).todense()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        savetxt(output_filename, all_data, fmt='%.3f', delimiter=',', header="#" * len(header))
        overwrite_header(header, output_filename)
    logging.info("Wrote fold {} to {} ({} instances {} MB)".format(nth_fold, output_filename, n_samples,
                                                                   get_size(output_filename)))


def overwrite_header(header, output_filename):
    """
    Fixes the header by re-writing the header in place using FileInput.

    :param header: the correct header
    :param output_filename:
    """
    with fileinput.FileInput(output_filename, inplace=1) as fi:
        fi.readline()
        print(header)
        for line in fi:
            print(line, end='')


if __name__ == "__main__":
    main()
