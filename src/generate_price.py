#!/usr/bin/env python

import logging
import os
import time
import datetime

import click

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from datasets.fuman_raw import load_fuman_csv, load_rants, load_target_rants
from datasets.csv_output import vector_headers, make_csv_row, save_dataset_metadata
from datasets.features import vectorize_text, vectorise_text_fit, encode_categoricals
from util.mecab import tokenize_rant, tokenize_pos, STOPWORDS
from util.file import get_size

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ALLSCORED = 'rants-price.csv'
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
def main(source, output, split_size, max_splits, word_max_features, pos_max_features, word_min_df, pos_min_df,
         pos_bad_only, word_vec, pos_vec, pos_ngram, encode):
    """
    Generates a price estimation training dataset from Fuman user posts. (Regression)

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

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    output_path = os.path.join(output, "price-" + timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.isfile(source):
        source_filepath = source
    else:
        source_filepath = os.path.join(source, ALLSCORED)
    pos_vec_func = vectorize_text
    if pos_bad_only:
        pos_vec_func = vectorise_text_fit
    word_vectorizer = VECTORIZERS[word_vec]
    pos_vectorizer = VECTORIZERS[pos_vec]
    logging.info("Loading vectors for pos and words")
    word_vects = vectorize_text(load_rants(filepath=source_filepath), None, vectorizer=TfidfVectorizer,
                                tokenizer=tokenize_rant, stop_words=STOPWORDS, min_df=word_min_df,
                                max_features=word_max_features)
    pos_vects = pos_vec_func(load_rants(filepath=source_filepath),
                             load_target_rants(filepath=source_filepath, target=1,
                                               target_var_func=set_price),
                             vectorizer=CountVectorizer, tokenizer=tokenize_pos, ngram_range=(1, pos_ngram),
                             min_df=pos_min_df, max_features=pos_max_features)
    if word_max_features is not 0 and pos_max_features is not 0:
        assert word_vects.shape[0] == pos_vects.shape[0], \
            "Word and Pos vector row counts dont match! w:{} p:{}".format(word_vects.shape[0], pos_vects.shape[0])
    logging.info("Loading instances from CSV...")
    instances = list(load_fuman_csv(source_filepath, target_var_func=set_price))
    if encode:
        instances = encode_categoricals(instances)
    n_instances = len(instances)
    if word_max_features is not 0:
        assert word_vects.shape[0] == n_instances, \
            "Instances and vector counts don't match! csv: %r vec: %r" % (n_instances, word_vects.shape[0])
    logging.info("OK! total:{}".format(n_instances))

    # output to CSV
    n_splits = 1
    split_start = 0
    headers = vector_headers(pos_vects, word_vects)
    n_columns = len(headers.split(','))
    logging.info("Final dataset: {} instances {} features".format(n_instances, n_columns))
    while n_splits <= max_splits and split_start < n_instances:
        output_filename = os.path.join(output_path, "{}-{}-{}.csv".format("price", timestamp, n_splits))
        logging.info("Writing to: " + output_filename)

        split_end = min(n_instances, split_start + split_size)
        with open(output_filename, 'w', encoding='utf-8') as out:
            out.write(headers)
            for i in range(split_start, split_end):
                row = make_csv_row(instances[i], i, pos_vects, word_vects)
                out.write(row)
        logging.info(
            "Wrote {} instances (total: {}) size:{} MB".format(split_end - split_start, split_end,
                                                               get_size(output_filename)))
        split_start = split_end
        n_splits += 1

    save_dataset_metadata(encode, output_path, "price", pos_max_features, pos_min_df, pos_ngram, pos_vec_func,
                          pos_vectorizer, source_filepath, timestamp, word_max_features, word_min_df, word_vectorizer,
                          tokenize_rant, tokenize_pos)
    logging.info("Work complete!")


def set_price(_, price):
    return int(price)


if __name__ == "__main__":
    main()
