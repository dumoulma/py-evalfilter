#!/usr/bin/env python
import logging
import os

import click

from sklearn.feature_extraction.text import TfidfVectorizer

from datasets.fuman_raw import load_fuman_csv, load_rants, get_header
from mecab import tokenize_pos, tokenize_rant, STOPWORDS

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ALLSCORED = 'all-scored-rants.csv'


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--split_size', default=1000000)
@click.option('--max_splits', default=2)
@click.option('--max_word_features', default=5000)
@click.option('--max_pos_features', default=5000)
@click.option('--word_min_df', default=100)
@click.option('--pos_min_df', default=100)
def main(source, output, split_size, max_splits, max_word_features, max_pos_features, word_min_df, pos_min_df):
    """
    Generates a training dataset from Fuman user posts.

    Concatenates simple features from the database, hand crafted features based on various character and word counts,
    and Tf-Idf weighted bag of words based on the text as well as the part-of-speech tags of Fuman user posts.

    :param source: directory of the input files. assumes the file names are the same as the dump script output.
    :param output: the output directory
    :param split_size: the size (in instances) of each split of the data
    :param max_splits: the number of splits to generate
    :param max_word_features: parameter for tf-idf vectorizer
    :param max_pos_features: parameter for tf-idf vectorizer
    :param word_min_df: parameter for tf-idf vectorizer
    :param pos_min_df: parameter for tf-idf vectorizer
    """
    if os.path.isfile(source):
        source_filepath = source
    else:
        source_filepath = os.path.join(source, ALLSCORED)
    wdvec = TfidfVectorizer(tokenizer=tokenize_rant, strip_accents='unicode', stop_words=STOPWORDS,
                            min_df=word_min_df, max_features=max_word_features)
    rants_vects = wdvec.fit_transform(load_rants(filepath=source_filepath))
    logging.info("Rants vectorized: {}".format(rants_vects.shape))
    posvec = TfidfVectorizer(tokenizer=tokenize_pos, ngram_range=(1, 3), strip_accents='unicode', min_df=pos_min_df,
                             max_features=max_pos_features)
    pos_vects = posvec.fit_transform(load_rants(filepath=source_filepath))
    logging.info("POS vectorized: {}".format(pos_vects.shape))
    logging.info("Shape: pos={} wd={}".format(pos_vects.shape, rants_vects.shape))

    instances = list(load_fuman_csv(source_filepath, target_var_func=set_goodvsbad_label))
    good_indices = list(i for i, x in enumerate(instances) if x[-1] is -1)
    bad_indices = list(i for i, x in enumerate(instances) if x[-1] is 1)

    # output to CSV
    split = 1
    n = 0
    m = 0
    n_instances = rants_vects.shape[0]
    n_bad = len(bad_indices)
    n_good = len(good_indices)
    while split <= max_splits and n < n_instances:
        output_filename = os.path.join(output, "{}-{}.csv".format("goodvsbad", split))
        logging.info("Writing to: " + output_filename)
        split_end = min(n_good, n + split_size)
        with open(output_filename, 'w', encoding='utf-8') as out:
            out.write(generate_header(pos_vects, rants_vects))
            for i in bad_indices:
                row = make_csv_row(instances[i], i, pos_vects, rants_vects)
                out.write(row)
            for i in range(m, split_end):
                row = make_csv_row(instances[good_indices[i]], i, pos_vects, rants_vects)
                out.write(row)
                m += 1
            n = m + (split - 1) * n_bad
        logging.info("Wrote {} instances (total: {})".format(n - (split - 1) * split_size, n))
        split += 1


def make_csv_row(x, i, pos_vects, rants_vects) -> str:
    features = ','.join(str(i) for i in x[:-1]) + ','
    pos = ','.join(str(j) for j in pos_vects[i].todense().tolist()[0]) + ','
    wd = ','.join(str(k) for k in rants_vects[i].todense().tolist()[0]) + ','
    row = features + pos + wd + str(x[-1]) + '\n'
    if i and i % 1000 == 0:
        logging.debug("{}:{}\t{}".format(i, row.split(',')[-1], row.split(',')[:25]))
        logging.debug("{}:{}".format(i, len(row.split(','))))
    return row


def generate_header(pos_vects, rants_vects):
    header = get_header()
    pos_header = generate_pos_headers(pos_vects)
    words_header = generate_word_headers(rants_vects)
    target_header = "target"
    final_header = ','.join((header, pos_header, words_header, target_header)) + '\n'
    logging.debug("Header: features:{} pos:{} wd:{} final:{}".format(len(header.split(',')),
                                                                     len(pos_header.split(',')),
                                                                     len(words_header.split(',')),
                                                                     len(final_header.split(','))))
    return final_header


def generate_pos_headers(pos_vect):
    return ','.join(["pos" + str(i) for i in range(pos_vect.shape[1])])


def generate_word_headers(word_vect):
    return ','.join(["wd" + str(i) for i in range(word_vect.shape[1])])


def encode_categoricals(X):
    """
    Transforms the categorical string values into int (necessary for some algorithms that can't process strings).
    :param X:
    :return:
    """
    import sklearn.preprocessing as pp
    age_enc = pp.LabelEncoder()
    encoded_age = age_enc.fit_transform([x[6] for x in X])
    state_enc = pp.LabelEncoder()
    encoded_state = state_enc.fit_transform([x[7] for x in X])
    job_enc = pp.LabelEncoder()
    encoded_job = job_enc.fit_transform([x[9] for x in X])
    for x, ea, es, ej in zip(X, encoded_age, encoded_state, encoded_job):
        x[6] = ea
        x[7] = es
        x[9] = ej


def categorical_to_binary(X):
    import sklearn.preprocessing as pp
    values = [len(set(x[6] for x in X)), len(set(x[6] for x in X)), len(set(x[6] for x in X))]
    ohe = pp.OneHotEncoder(n_values=values, categorical_features=[6, 7, 9], sparse=False)
    return ohe.fit_transform(X)


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
