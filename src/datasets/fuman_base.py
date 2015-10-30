import logging
import csv
from os import path
from datetime import date

from sklearn.datasets.base import Bunch
import sklearn.preprocessing as pp
import numpy as np

import unicodedata

FUMAN_PATH = "../../data/20150930/good-vs-bad-100.csv"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

GENDER = {0: "unk", 1: 'male', 2: 'female'}
BOOLEAN = {0: 'False', 1: 'True'}


def get_gender(raw_gender):
    if raw_gender == '''\\0''':
        g = 0
    else:
        try:
            g = int(raw_gender)
        except ValueError as ve:
            logging.warning("Can't parse gender, set to UNKNOWN (got: {})".format(ve))
            g = 0
    return GENDER[g]


def get_age(raw_age):
    age = int(raw_age)
    if age is 0:
        return 0
    return date.today().year - age


def to_binary_categorical(raw_field):
    return BOOLEAN[int(raw_field)]


def check_row_format(i, row):
    if not isinstance(row, list):
        row = row.rstrip().split(',')
    if len(row) is not 16:
        logging.debug("Row with bad number of fields at line {}".format(i))
        return False
    try:
        int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[6]), int(row[7]), int(row[8]), int(row[15])
        return True
    except ValueError as ve:
        logging.debug("Parse problem for row {}: {} ({})".format(i, row, ve))
    return False


def load_fuman_rants2(file_path, target_func):
    good_data = load_fuman_rants(path.join(file_path, "good-rants.csv"), target_func)
    bad_data = load_fuman_rants(path.join(file_path, "bad-rants.csv"), target_func)
    return Bunch(data=good_data.data.append(bad_data.data),
                 target=good_data.target.append(bad_data.target),
                 DESCR="Fuman DB csv dump dataset")


def load_fuman_rants(file_path, target_func):
    # with open(file_path, newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',', quotechar="'")
    #     next(reader)  # skip headers
    #     n_samples = sum(1 for i, row in enumerate(csvfile) if check_row_format(i, row)) + 1
    data = list()
    target = list()
    parse_errors = 0
    n_samples = 0
    with open(file_path, newline='') as csv_file:
        data_file = csv.reader(csv_file, delimiter=',', quotechar="'")
        next(data_file)
        for row in data_file:
            if not check_row_format(row[0], row):
                parse_errors += 1
                continue
            data.append(unicodedata.normalize('NFKC', row[5]))
            status = int(row[6])
            price = int(row[15])
            target.append(target_func(status, price))
            n_samples += 1
    logging.info('Finished loading data. (read: {} errors: {})'.format(n_samples, parse_errors))
    return Bunch(data=data,
                 target=target,
                 DESCR="Fuman DB csv dump dataset")


def load_fuman_gvb(file_path, bad_filename="bad-rants.csv", good_filename="good-rants.csv"):
    data = list()
    target = list()
    parse_errors = 0
    n_read = 0
    with open(path.join(file_path, bad_filename), newline='') as csv_file:
        data_file = csv.reader(csv_file, delimiter=',', quotechar="'")
        next(data_file)
        for row in data_file:
            data.append(unicodedata.normalize('NFKC', row[5]))
            target.append(1)
            n_read += 1
    n_bad = n_read
    logging.info("Read {} bad instances".format(n_read))
    with open(path.join(file_path, good_filename), newline='') as csv_file:
        data_file = csv.reader(csv_file, delimiter=',', quotechar="'")
        next(data_file)
        for row in data_file:
            if not check_row_format(row[0], row):
                parse_errors += 1
                continue
            data.append(unicodedata.normalize('NFKC', row[5]))
            target.append(-1)
            n_read += 1
    logging.info("Read {} good instances".format(n_read - n_bad))
    logging.info('Finished loading data. (read: {} errors: {})'.format(n_read, parse_errors))
    return Bunch(data=data,
                 target=target,
                 DESCR="Fuman DB csv dump dataset")


def fuman_gvb_target(status, _):
    if status is 100:  # code 100 is good fuman post
        return -1
    elif 200 <= status < 300:  # code 2XX are for bad fuman posts
        return 1
    else:
        raise ValueError("Unexpected value for status")


def fuman_price_target(_, price):
    return int(price)


def load_fuman(file_path):
    data, target = zip(*[(r[:-1], r[-1]) for r in FumanDataset(file_path)])
    logging.info("Finished reading data")
    data, target = list(data), list(target)
    le1 = pp.LabelEncoder()
    encoded_gender = le1.fit_transform([x[23] for x in data])
    le2 = pp.LabelEncoder()
    encoded_state = le2.fit_transform([x[23] for x in data])
    le3 = pp.LabelEncoder()
    encoded_job = le3.fit_transform([x[23] for x in data])
    for x, eg, es, ej in zip(data, encoded_gender, encoded_state, encoded_job):
        x[23] = eg
        x[25] = es
        x[26] = ej
    return Bunch(data=data,
                 target=target,
                 target_names=np.arange(10),
                 DESCR="Fuman DB csv dump dataset")


class FumanDataset(object):
    """
        Streams through the rants from the Fuman DB dump CSV file found
        in 'filename', up to a maximum of k posts.

        usage: for post in rant_iter(filename='path/to/file.csv')
        :returns post (Str)
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar="'")
            headers = next(reader)  # skip headers
            logging.info("Got headers: {}".format(headers))
            for row in reader:
                yield row
