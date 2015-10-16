#!/usr/bin/env python

import logging
import os
import time
import datetime
import json

import click

from datasets.fuman_raw import load_fuman_csv, load_rants, load_target_rants
from datasets.csv_output import generate_header, make_csv_row
from datasets.features import tfidf_word, bad_pos_vects
from util.mecab import tokenize_rant, tokenize_pos, STOPWORDS

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ALLSCORED = 'all-scored-rants.csv'
POS_NGRAM_RANGE = (1, 3)


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--split_size', default=10000)
@click.option('--max_splits', default=2)
@click.option('--word_max_features', default=5000)
@click.option('--pos_max_features', default=5000)
@click.option('--word_min_df', default=100)
@click.option('--pos_min_df', default=100)
def main(source, output, split_size, max_splits, word_max_features, pos_max_features, word_min_df, pos_min_df):
    """
    Generates a good vs bad training dataset from Fuman user posts. Binary Classification.

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
    if not os.path.isdir(output):
        raise ValueError("Output must be a directory")

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    output_path = os.path.join(output, timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.isfile(source):
        source_filepath = source
    else:
        source_filepath = os.path.join(source, ALLSCORED)

    logging.info("Loading vectors for pos and words")
    word_vects = tfidf_word(load_rants(filepath=source_filepath), tokenize_rant, STOPWORDS, word_min_df,
                            word_max_features)
    pos_vects = bad_pos_vects(load_rants(filepath=source_filepath),
                              load_target_rants(filepath=source_filepath, target=1,
                                                target_var_func=set_goodvsbad_label),
                              tokenize_pos, POS_NGRAM_RANGE, pos_min_df, pos_max_features)

    assert word_vects.shape[0] == pos_vects.shape[0], \
        "Word and Pos vector rows dont match! w:{} p:{}".format(word_vects.shape[0], pos_vects.shape[0])
    logging.info("Loading instances from CSV...")
    instances = list(load_fuman_csv(source_filepath, target_var_func=set_goodvsbad_label))
    good_indices = list(i for i, x in enumerate(instances) if x[-1] is -1)
    bad_indices = list(i for i, x in enumerate(instances) if x[-1] is 1)
    n_bad = len(bad_indices)
    n_good = len(good_indices)
    n_instances = pos_vects.shape[0]
    assert n_instances == n_good + n_bad, \
        "Number of instances don't match! csv: %r vec: %r" % (n_good + n_bad, n_instances)
    logging.info("OK! good:{} bad:{} total:{}".format(n_good, n_bad, n_instances))

    # output to CSV
    split = 1
    n = 0
    m = 0
    while split <= max_splits and n < n_instances:
        output_filename = os.path.join(output_path, "{}-{}-{}.csv".format("goodvsbad", timestamp, split))
        logging.info("Writing to: " + output_filename)
        split_end = min(n_good, n + split_size - split * n_bad)
        with open(output_filename, 'w', encoding='utf-8') as out:
            headers = generate_header(pos_vects, word_vects)
            n_columns = len(headers.split(','))
            out.write(headers)
            for i in bad_indices:
                row = make_csv_row(instances[i], i, pos_vects, word_vects)
                assert n_columns == len(row.split(',')), \
                    "row columns doesn'm match header! h:{} r:{}".format(n_columns, len(row.split(',')))
                out.write(row)
            split_start = m
            for i in range(split_start, split_end):
                row = make_csv_row(instances[good_indices[i]], i, pos_vects, word_vects)
                out.write(row)
                m += 1
        n = m + split * n_bad
        logging.info("Wrote {} instances (total: {})".format(split_end - split_start + n_bad, n))
        split += 1

    dataset_meta = {
        'timestamp': timestamp,
        'word_max_features': str(word_max_features),
        'pos_max_features': str(pos_max_features),
        'word_min_df': str(word_min_df),
        'pos_min_df': str(pos_min_df),
        'pos_ngram_range': str(POS_NGRAM_RANGE),
    }
    metadata_output = os.path.join(output_path, "metadata-{}.json".format(timestamp))
    with open(metadata_output, 'w', encoding='utf-8') as out:
        out.write(json.dumps(dataset_meta, indent=4, separators=(',', ': ')))
    logging.info("Metadata saved to {}".format(metadata_output))
    logging.info("Work complete!")


def set_goodvsbad_label(status, _):
    if status is 100:  # code 100 is good fuman post
        return -1
    elif 200 <= status < 300:  # code 2XX are for bad fuman posts
        return 1
    else:
        raise ValueError("Unexpected value for status")


if __name__ == "__main__":
    main()
