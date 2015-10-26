#!/usr/bin/env python

import logging
import os
import time
import datetime
import random
import json

import click

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from datasets.fuman_raw import load_fuman_csv, load_rants, load_target_rants, manual_features_header
from datasets.csv_output import vector_headers, make_csv_row, save_dataset_metadata, \
    make_svmlight_row
from datasets.features import vectorize_text, vectorise_text_fit, encode_categoricals
from util.mecab import tokenize_rant, tokenize_pos, STOPWORDS
from util.file import get_size

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ALLSCORED = 'all-scored-rants.csv'
VECTORIZERS = {'tfidf': TfidfVectorizer, 'count': CountVectorizer}


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--split_size', default=10000)
@click.option('--max_splits', default=2)
@click.option('--word_max_features', default=5000)
@click.option('--pos_max_features', default=5000)
@click.option('--pos_bad_only', is_flag=True)
@click.option('--word_min_df', default=100)
@click.option('--pos_min_df', default=100)
@click.option('--pos_ngram', default=3)
@click.option('--word_vec', type=click.Choice(['tfidf', 'count']), default='tfidf')
@click.option('--pos_vec', type=click.Choice(['tfidf', 'count']), default='count')
@click.option('--encode', is_flag=True)
@click.option('--svmlight', is_flag=True)
@click.option('--simple_headers', is_flag=True)
def main(source, output, split_size, max_splits, word_max_features, pos_max_features, word_min_df, pos_min_df,
         pos_bad_only, word_vec, pos_vec, pos_ngram, encode, svmlight, simple_headers):
    """
    Generates a good vs bad training dataset from Fuman user posts. (Binary Classification)

    Concatenates simple features from the database, hand crafted features based on various character and word counts,
    and Tf-Idf weighted bag of words based on the text as well as the part-of-speech tags of Fuman user posts.

    :param source: directory or file of the input files. (If dir, file will be all-scored-rants.csv)
    :param output: the output directory
    :param split_size: the size (in instances) of each n_splits of the data
    :param max_splits: the number of splits to generate
    :param word_max_features: parameter for tf-idf vectorizer (default 50000)
    :param pos_max_features: parameter for tf-idf vectorizer (default 50000)
    :param word_min_df: parameter for tf-idf vectorizer (default 100)
    :param pos_min_df: parameter for tf-idf vectorizer (default 100)
    :param pos_bad_only: learn vocabulary for POS from bad rants only (flag, default is all dataset)
    :param word_vec: [tfidf, count] use corresponding term weighting
    :param pos_vec: [tfidf, count] use corresponding term weighting
    :param pos_ngram: Learn vocabulary with ngrams in range (1,pos_ngram) (default is 3)
    :param encode:
    """
    if not os.path.isdir(output):
        raise ValueError("Output must be a directory")
    if svmlight and not encode:
        raise ValueError("smlight option set, but encode option not set!")

    if os.path.isfile(source):
        source_filepath = source
    else:
        source_filepath = os.path.join(source, ALLSCORED)
    logging.info("Source dump: {}".format(source_filepath))
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    logging.info("Timestamp: {}".format(timestamp))
    output_path = os.path.join(output, "gvb-" + timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pos_vec_func = vectorize_text
    if pos_bad_only:
        pos_vec_func = vectorise_text_fit
    word_vectorizer = VECTORIZERS[word_vec]
    pos_vectorizer = VECTORIZERS[pos_vec]
    word_dict_filename = os.path.join(output_path, "word-vocabulary-" + timestamp + ".json")
    pos_dict_filename = os.path.join(output_path, "pos-vocabulary-" + timestamp + ".json")

    logging.info("Loading vectors for pos and words")
    word_vects, word_features = vectorize_text(load_rants(filepath=source_filepath), None, vectorizer=TfidfVectorizer,
                                               tokenizer=tokenize_rant, stop_words=STOPWORDS, min_df=word_min_df,
                                               max_features=word_max_features)
    pos_vects, pos_features = pos_vec_func(load_rants(filepath=source_filepath),
                                           load_target_rants(filepath=source_filepath, target=1,
                                                             target_var_func=set_goodvsbad_label),
                                           vectorizer=CountVectorizer, tokenizer=tokenize_pos,
                                           ngram_range=(1, pos_ngram),
                                           min_df=pos_min_df, max_features=pos_max_features)

    save_features_json(word_dict_filename, word_features)
    save_features_json(pos_dict_filename, pos_features)

    if word_max_features is not 0 and pos_max_features is not 0:
        assert word_vects.shape[0] == pos_vects.shape[0], \
            "Word and Pos vector row counts dont match! w:{} p:{}".format(word_vects.shape[0], pos_vects.shape[0])
    logging.info("Loading instances from CSV...")
    instances = list(load_fuman_csv(source_filepath, target_var_func=set_goodvsbad_label))
    if encode:
        instances = encode_categoricals(instances)
    write_dataset(instances, max_splits, output_path, pos_features, pos_vects, simple_headers, split_size, svmlight,
                  timestamp, word_features, word_max_features, word_vects)

    save_dataset_metadata(encode, output_path, "goodvsbad", pos_max_features, pos_min_df, pos_ngram, pos_vec_func,
                          pos_vectorizer, source_filepath, timestamp, word_max_features, word_min_df, word_vectorizer,
                          tokenize_rant, tokenize_pos)
    logging.info("Work complete!")


def write_dataset(instances, max_splits, output_path, pos_features, pos_vects, simple_headers, split_size, svmlight,
                  timestamp, word_features, word_max_features, word_vects):
    good_indices = list(i for i, x in enumerate(instances) if x[-1] is -1)
    bad_indices = list(i for i, x in enumerate(instances) if x[-1] is 1)
    # makes the data I.I.D.
    random.shuffle(good_indices)
    random.shuffle(bad_indices)
    n_bad = len(bad_indices)
    n_good = len(good_indices)
    n_instances = n_good + n_bad
    if word_max_features is not 0:
        assert word_vects.shape[0] == n_instances, \
            "Instances and vector counts don't match! csv: %r vec: %r" % (n_instances, word_vects.shape[0])
    logging.info("OK! good:{} bad:{} total:{}".format(n_good, n_bad, n_instances))
    # output to CSV
    split = 1
    n_written = 0
    headers = manual_features_header()
    headers += vector_headers(pos_features, word_features, simple=simple_headers)
    headers += ",target\n"
    n_columns = len(headers.split(','))
    logging.info("Final dataset: {} instances {} features".format(n_instances, n_columns))
    while split <= max_splits and n_written < n_instances:
        logging.debug("split {} max_splits {} n_written {} n_instances {}".format(split, max_splits,
                                                                                  n_written, n_instances))
        output_filename = os.path.join(output_path, "{}-{}-{}.csv".format("goodvsbad", timestamp, split))
        logging.info("Writing to: " + output_filename)
        with open(output_filename, 'w', encoding='utf-8') as out:
            if not svmlight:
                out.write(headers)
            for i in bad_indices:
                row = next_row(i, instances, n_columns, pos_vects, word_vects, svmlight)
                out.write(row)
            n_written += n_bad
            logging.debug("split: {} after bad: {} written".format(split, n_written))
            while len(good_indices) is not 0 and n_written % split_size is not 0:
                i = good_indices.pop()
                row = next_row(i, instances, n_columns, pos_vects, word_vects, svmlight)
                out.write(row)
                n_written += 1
            logging.debug("split: {} after good: {} written".format(split, n_written))
        split += 1
        n_written_split = split_size if n_written % split_size is 0 else n_written % split_size
        logging.info("Wrote {} instances (total: {}) size:{} MB".format(n_written_split, n_written,
                                                                        get_size(output_filename)))


def next_row(i, instances, n_columns, pos_vects, word_vects, svmlight):
    if svmlight:
        row = make_svmlight_row(instances[i][:-1], instances[i][-1], i, pos_vects, word_vects)
    else:
        row = make_csv_row(instances[i], i, pos_vects, word_vects)
        assert n_columns == len(row.split(',')), \
            "row columns doesn'm match header! h:{} r:{}".format(n_columns, len(row.split(',')))
    return row


def save_features_json(filepath, feature_names):
    if not feature_names:
        return
    with open(filepath, mode='w') as out:
        out.write(json.dumps(feature_names, ensure_ascii=False, indent=4, separators=(',', ':')))
        logging.info("Saved features to: {}".format(filepath))


def set_goodvsbad_label(status, _):
    if status is 100:  # code 100 is good fuman post
        return -1
    elif 200 <= status < 300:  # code 2XX are for bad fuman posts
        return 1
    else:
        raise ValueError("Unexpected value for status")


if __name__ == "__main__":
    main()
