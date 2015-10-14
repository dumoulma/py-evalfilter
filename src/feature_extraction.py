#!/usr/bin/env python
import logging
import sys

import click

from sklearn.feature_extraction.text import TfidfVectorizer

from datasets.fuman_raw import load_fuman_csv, load_rants, get_header
from mecab import tokenize_pos, tokenize_rant, STOPWORDS

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
@click.argument('output', type=click.Path(), nargs=1)
@click.option('--split_size', default=sys.maxsize)
@click.option('--max_word_features', default=5000)
@click.option('--max_pos_features', default=5000)
@click.option('--word_min_df', default=100)
@click.option('--pos_min_df', default=100)
def main(source, output, split_size, max_word_features, max_pos_features, word_min_df, pos_min_df):
    wdvec = TfidfVectorizer(tokenizer=tokenize_rant, strip_accents='unicode', stop_words=STOPWORDS,
                            min_df=word_min_df, max_features=max_word_features)
    rants_vects = wdvec.fit_transform(load_rants(filepath=source))
    logging.info("Rants vectorized: {}".format(rants_vects.shape))
    posvec = TfidfVectorizer(tokenizer=tokenize_pos, ngram_range=(1, 3), strip_accents='unicode', min_df=pos_min_df,
                             max_features=max_pos_features)
    pos_vects = posvec.fit_transform(load_rants(filepath=source))
    logging.info("POS vectorized: {}".format(pos_vects.shape))

    X = list(load_fuman_csv(source, target_var_func=set_goodvsbad_label))
    print("{} negatives found".format(abs(sum(filter(lambda label: label is 1, [x[-1] for x in X])))))
    logging.info("Got {} rows".format(len(X)))
    logging.info("x={} pos={} wd={}".format(len(X[0]), pos_vects.shape[1], rants_vects.shape[1]))

    # output to CSV
    split = 1
    n = 0
    n_instances = rants_vects.shape[0]
    while n < n_instances:
        output_filename = output + "-" + str(split)
        logging.info("Writing to: " + output_filename)
        next_start = n
        with open(output_filename, 'w', encoding='utf-8') as out:
            out.write(generate_header(pos_vects, rants_vects))
            for i in range(next_start, min(n_instances, next_start + split_size)):
                x = X[i]
                features = ','.join(str(i) for i in x[:-1]) + ','
                pos = ','.join(str(j) for j in pos_vects[i].todense().tolist()[0]) + ','
                wd = ','.join(str(k) for k in rants_vects[i].todense().tolist()[0]) + ','
                row = features + pos + wd + str(x[-1]) + '\n'
                if i and i % 1000 == 0:
                    logging.info("{}:{}".format(i, len(row.split(','))))
                out.write(row)
                n += 1
            logging.info("Wrote {} instances".format(n - next_start))
            split += 1


def generate_header(pos_vects, rants_vects):
    header = get_header()
    pos_header = generate_pos_headers(pos_vects)
    words_header = generate_word_headers(rants_vects)
    target_header = "target"
    final_header = ','.join((header, pos_header, words_header, target_header)) + '\n'
    logging.info("Header: features:{} pos:{} wd:{} final:{}".format(len(header.split(',')),
                                                                    len(pos_header.split(',')),
                                                                    len(words_header.split(',')),
                                                                    len(final_header.split(','))))
    return final_header


def generate_pos_headers(pos_vect):
    return ','.join(["pos" + str(i) for i in range(pos_vect.shape[1])])


def generate_word_headers(word_vect):
    return ','.join(["wd" + str(i) for i in range(word_vect.shape[1])])


# def encode_categoricals(X):
#     age_enc = pp.LabelEncoder()
#     encoded_age = age_enc.fit_transform([x[6] for x in X])
#     state_enc = pp.LabelEncoder()
#     encoded_state = state_enc.fit_transform([x[7] for x in X])
#     job_enc = pp.LabelEncoder()
#     encoded_job = job_enc.fit_transform([x[9] for x in X])
#     for x, ea, es, ej in zip(X, encoded_age, encoded_state, encoded_job):
#         x[6] = ea
#         x[7] = es
#         x[9] = ej


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
