#!/usr/bin/env python

import logging
import os
import time
import datetime
import warnings

import click
import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn.cross_validation import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline

from datasets.fuman_base import load_fuman_gvb
from datasets.output import save_dataset_metadata, save_features_json, make_header, dump_csv
from util.mecab import tokenize_pos
from datasets.fuman_features import RantStats

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

GOOD_FILENAME = "good-rants-4189.csv"
BAD_FILENAME = "bad-rants-4189.csv"
PRICE_FILENAME = "rants-price.csv"
VECTORIZERS = {'tfidf': TfidfVectorizer, 'count': CountVectorizer}


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--n_folds', default=5)
@click.option('--n_folds_max', default=2)
@click.option('--pos_max_features', default=3000)
@click.option('--pos_min_df', default=25)
@click.option('--pos_ngram', default=2)
@click.option('--pos_vec', type=click.Choice(['tfidf', 'count']), default='count')
@click.option('--sparse', is_flag=True)
@click.option('--feature_name_header', is_flag=True)
def main(source, output, n_folds, n_folds_max, pos_max_features, pos_min_df, pos_vec, pos_ngram,
         sparse, feature_name_header):
    """
    Generates a good vs bad training dataset from Fuman user posts. (Binary Classification)

    Concatenates simple features from the database, hand crafted features based on various character and word counts,
    and Tf-Idf weighted bag of words based on the text as well as the part-of-speech tags of Fuman user posts.

    :param source: directory or file of the input files. (If dir, file will be all-scored-rants.csv)
    :param output: the output directory
    :param n_folds: the number of splits to generate (using StratifiedKFold)
    :param pos_max_features: parameter for tf-idf vectorizer (default 50000)
    :param pos_min_df: parameter for tf-idf vectorizer (default 100)
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
    output_path = os.path.join(output, "gvsb-{}".format(timestamp))

    rant_stats_vectorizer = DictVectorizer()
    pipeline = Pipeline([('stats', RantStats()),  # returns a list of dicts
                         ('vect', rant_stats_vectorizer)])  # list of dicts -> feature matrix
    vectorizer = None
    pos_dict_filename = os.path.join(output_path, "pos-vocabulary-" + timestamp + ".json")
    if pos_max_features:
        pos_vectorizer_func = VECTORIZERS[pos_vec]
        vectorizer = pos_vectorizer_func(tokenizer=tokenize_pos, ngram_range=(1, pos_ngram), strip_accents='unicode',
                                         min_df=pos_min_df, max_features=pos_max_features)
        pipeline = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[
                    ('rant_stats', pipeline),
                    ("pos_vec", vectorizer),
                ]))
        ])

    fuman_data = load_fuman_gvb(source, good_filename=GOOD_FILENAME, bad_filename=BAD_FILENAME)
    logging.info("Processing pipeline...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="deprecated", module="sklearn")
        instances = pipeline.fit_transform(fuman_data.data)
        n_samples = instances.shape[0]
        y = np.asarray(fuman_data.target, dtype=np.int8).reshape((n_samples,))

        pos_features = list()
        if pos_max_features:
            pos_features = vectorizer.get_feature_names()
            save_features_json(pos_dict_filename, pos_features)
        header = make_header(rant_stats_vectorizer.get_feature_names(), pos_features=pos_features,
                             feature_name_header=feature_name_header)
        logging.info("Saving {} folds to disk...".format(n_folds))
        skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
        for i, (_, test_index) in enumerate(skf, 1):
            dump_csv(output_path, instances[test_index], y[test_index], i, header, timestamp, sparse)
            if i == n_folds_max:
                break
        save_dataset_metadata(sparse, output_path, "goodvsbad", source_filepath=source, timestamp=timestamp,
                              pos_vectorizer=vectorizer, tokenize_pos=tokenize_pos)
    logging.info("Work complete!")


def get_pos_header(features, simple_headers):
    headers = features
    if simple_headers:
        headers = ["pos_" + str(i) for i in range(len(features))]
    return ','.join(headers)


if __name__ == "__main__":
    main()
