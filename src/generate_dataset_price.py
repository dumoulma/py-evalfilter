#!/usr/bin/env python

import logging
import os
import time
import datetime
import warnings

import click
import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn.cross_validation import KFold

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline

from datasets import load_fuman_price
from evalfilter.analysis import tokenize_pos, tokenize_rant, tokenize_token_type
from evalfilter import FieldSelector, RantStats, UserProfileStats, save_dataset_metadata, save_features_json, \
    make_header, dump_csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

GOOD_FILENAME = "good-rants.csv"
BAD_FILENAME = "bad-rants.csv"
PRICE_FILENAME = "rants-price.csv"
VECTORIZERS = {'tfidf': TfidfVectorizer, 'count': CountVectorizer}


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--n_folds', default=3)
@click.option('--n_folds_max', default=1)
@click.option('--pos_max_features', default=3000)
@click.option('--pos_min_df', default=25)
@click.option('--type_max_features', default=1000)
@click.option('--type_min_df', default=10)
@click.option('--type_ngram', default=3)
@click.option('--word_max_features', default=0)
@click.option('--word_min_df', default=25)
@click.option('--pos_ngram', default=2)
@click.option('--pos_vec', type=click.Choice(['tfidf', 'count']), default='count')
@click.option('--type_vec', type=click.Choice(['tfidf', 'count']), default='count')
@click.option('--sparse', is_flag=True)
@click.option('--feature_name_header', is_flag=True)
def main(source, output, n_folds, n_folds_max, word_max_features, word_min_df, pos_max_features, pos_min_df,
         pos_vec, pos_ngram, type_max_features, type_min_df, type_vec, type_ngram, sparse, feature_name_header):
    """
    Generates a good vs bad training dataset from Fuman user posts. (Binary Classification)

    Concatenates simple features from the database, hand crafted features based on various character and word counts,
    and Tf-Idf weighted bag of words based on the text as well as the part-of-speech tags of Fuman user posts.

    :param source: directory or file of the input files. (If dir, file will be all-scored-rants.csv)
    :param output: the output directory
    :param n_folds: the number of splits to generate (using StratifiedKFold)
    :param n_folds_max: max number of folds to output
    :param pos_max_features: parameter for tf-idf vectorizer of POS (default 3000)
    :param pos_min_df: parameter for tf-idf vectorizer of POS (default 25)
    :param word_max_features: parameter for tf-idf vectorizer of words (default 3000)
    :param word_min_df: parameter for tf-idf vectorizer of words (default 25)
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

    logging.info("Timestamp: {}".format(timestamp))
    pos_dict_filename = os.path.join(output_path, "pos-features-" + timestamp + ".json")
    type_dict_filename = os.path.join(output_path, "type-features-" + timestamp + ".json")
    rant_dict_filename = os.path.join(output_path, "rant-features-" + timestamp + ".json")

    rant_stats_vectorizer = DictVectorizer()
    userprofile_vectorizer = DictVectorizer()
    transformer_list = [
        ('rant_stats', Pipeline([
            ('selector', FieldSelector(key='rant')),
            ('stats', RantStats()),  # returns a list of dicts
            ('vect', rant_stats_vectorizer),  # list of dicts -> feature matrix
        ])),
        ('userprofile_stats', Pipeline([
            ('selector', FieldSelector(key='userprofile')),
            ('stats', UserProfileStats()),  # returns a list of dicts
            ('vect', userprofile_vectorizer),  # list of dicts -> feature matrix
        ])),
    ]

    pos_vectorizer_func = VECTORIZERS[pos_vec]
    pos_vectorizer = None
    type_vectorizer_func = VECTORIZERS[type_vec]
    type_vectorizer = None
    word_vectorizer = None
    if pos_max_features:
        pos_vectorizer = pos_vectorizer_func(tokenizer=tokenize_pos, ngram_range=(1, pos_ngram),
                                             strip_accents='unicode', min_df=pos_min_df, max_features=pos_max_features)
        transformer_list.append(('pos_bow', Pipeline([
            ('selector', FieldSelector(key='rant')),
            ('vectorize', pos_vectorizer),
        ])))
    if type_max_features:
        type_vectorizer = type_vectorizer_func(tokenizer=tokenize_token_type, ngram_range=(1, type_ngram),
                                               strip_accents='unicode', min_df=type_min_df,
                                               max_features=type_max_features)
        transformer_list.append(('type_bow', Pipeline([
            ('selector', FieldSelector(key='rant')),
            ('vectorize', type_vectorizer),
        ])))
    if word_max_features:
        word_vectorizer = TfidfVectorizer(tokenizer=tokenize_rant, strip_accents='unicode', min_df=word_min_df,
                                          max_features=word_max_features)
        transformer_list.append(('rant_bow', Pipeline([
            ('selector', FieldSelector(key='rant')),
            ('vectorize', word_vectorizer),
        ])))
    pipeline = Pipeline([
        ('union', FeatureUnion(transformer_list=transformer_list))
    ])

    fuman_data = load_fuman_price(source_dir, filename=source_filename)
    logging.info("Processing pipeline...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="sklearn")
        instances = pipeline.fit_transform(fuman_data.data)
        n_samples = instances.shape[0]
        y = np.asarray(fuman_data.target, dtype=np.int8).reshape((n_samples,))

        pos_features = list()
        type_features = list()
        rant_features = list()
        if pos_max_features:
            pos_features = pos_vectorizer.get_feature_names()
            save_features_json(pos_dict_filename, pos_features)
        if type_max_features:
            type_features = type_vectorizer.get_feature_names()
            save_features_json(type_dict_filename, type_features)
        if word_max_features:
            rant_features = word_vectorizer.get_feature_names()
            save_features_json(rant_dict_filename, rant_features)
        header = make_header(rant_stats_vectorizer.get_feature_names(),
                             userprofile_vectorizer.get_feature_names(),
                             pos_features, type_features, rant_features, feature_name_header)
        logging.info("Saving {} of {} folds to disk...".format(n_folds_max, n_folds))
        if n_folds == 1:
            dump_csv(output_path, instances, y, 0, header, timestamp, sparse)
        else:
            skf = KFold(n=n_samples, n_folds=n_folds, shuffle=True)
            for i, (_, test_index) in enumerate(skf, 1):
                dump_csv(output_path, instances[test_index], y[test_index], "price", i, header, timestamp, sparse)
                if i == n_folds_max:
                    break
        save_dataset_metadata(sparse, output_path, "price", source_filepath=source, timestamp=timestamp,
                              word_vectorizer=word_vectorizer, tokenize_rant=tokenize_rant,
                              pos_vectorizer=pos_vectorizer, tokenize_pos=tokenize_pos, type_vectorizer=type_vectorizer,
                              tokenize_type=tokenize_token_type)
    logging.info("Work complete!")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="getargspec", category=DeprecationWarning)
        main()
