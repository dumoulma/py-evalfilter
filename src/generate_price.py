#!/usr/bin/env python

import logging
import os
import time
import datetime

import click

from datasets.fuman_raw import load_fuman_csv
from datasets.csv_output import generate_header, make_csv_row
from datasets.features import tfidf_word, tfidf_pos
from mecab import tokenize_rant, tokenize_pos, STOPWORDS

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ALLSCORED = 'all-scored-rants.csv'


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--split_size', default=1000000)
@click.option('--max_splits', default=2)
@click.option('--word_max_features', default=5000)
@click.option('--pos_max_features', default=5000)
@click.option('--word_min_df', default=100)
@click.option('--pos_min_df', default=100)
def main(source, output, split_size, max_splits, word_max_features, pos_max_features, word_min_df, pos_min_df):
    """
    Generates a price prediction training dataset from Fuman user posts. Regression.

    Concatenates simple features from the database, hand crafted features based on various character and word counts,
    and Tf-Idf weighted bag of words based on the text as well as the part-of-speech tags of Fuman user posts.

    :param source: directory or file of the input files. (If dir, file will be all-scored-rants.csv)
    :param output: the output directory
    :param split_size: the size (in instances) of each split of the data
    :param max_splits: the number of splits to generate
    :param word_max_features: parameter for tf-idf vectorizer
    :param pos_max_features: parameter for tf-idf vectorizer
    :param word_min_df: parameter for tf-idf vectorizer
    :param pos_min_df: parameter for tf-idf vectorizer
    """
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    if os.path.isdir(output):
        output_path = os.path.join(output, timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.isfile(source):
        source_filepath = source
    else:
        source_filepath = os.path.join(source, ALLSCORED)

    logging.info("Loading vectors for pos and words")
    word_vects = tfidf_word(source_filepath, tokenize_rant, STOPWORDS, word_min_df, word_max_features)
    pos_vects = tfidf_pos(source_filepath, tokenize_pos, (1, 3), pos_min_df, pos_max_features)

    logging.info("Loading instances...")
    instances = list(load_fuman_csv(source_filepath, target_var_func=set_price))
    n_instances = word_vects.shape[0]
    assert n_instances is len(instances), \
        "Number of instances don't match! csv: %r vec: %r" % (len(instances), n_instances)
    logging.info("OK! total:{}".format(n_instances))

    # output to CSV
    split = 1
    split_start = 0
    while split <= max_splits and split_start < n_instances:
        output_filename = os.path.join(output_path, "{}-{}-{}.csv".format("goodvsbad", timestamp, split))
        logging.info("Writing to: " + output_filename)
        split_end = min(n_instances, split_start + split_size)
        with open(output_filename, 'w', encoding='utf-8') as out:
            out.write(generate_header(pos_vects, word_vects))
            for i, x in enumerate(instances):
                row = make_csv_row(x, i, pos_vects, word_vects)
                out.write(row)
        logging.info("Wrote {} instances (total: {})".format(split_end - split_start, split_end))
        split_start = split_end
        split += 1


def set_price(_, price):
    return price


if __name__ == "__main__":
    main()
