import logging
import csv
from os import path
from datetime import date
from collections import defaultdict

from sklearn.datasets.base import Bunch

import unicodedata

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
    if len(row) is not 16 and len(row) is not 15:
        logging.debug("Row with bad number of fields at line {}".format(i))
        return False
    try:
        int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[6]), int(row[7]), int(row[8])
        if len(row) is 16:
            int(row[15])
        return True
    except ValueError as ve:
        logging.debug("Parse problem for row {}: {} ({})".format(i, row, ve))
    return False


def fuman_price_target(_, price):
    return int(price)


def fuman_gvb_target(status, _):
    if status is 100:  # code 100 is good fuman post
        return -1
    elif 200 <= status < 300:  # code 2XX are for bad fuman posts
        return 1
    else:
        raise ValueError("Unexpected value for status")


def load_fuman_price(file_path, filename="rants-price.csv"):
    source_path = path.join(file_path, filename)
    return load_fuman_userprofile(source_path, fuman_price_target)


def load_fuman_userprofile(file_path, target_func):
    data = defaultdict(list)
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
            x = dict()
            x['hasindustry'] = to_binary_categorical(row[1])
            x['hasoccupation'] = to_binary_categorical(row[2])
            x['hascompany'] = to_binary_categorical(row[3])
            x['hasprodname'] = to_binary_categorical(row[4])
            x['hasproposal'] = to_binary_categorical(row[7])
            x['empathies'] = int(row[8])
            x['birthyear'] = get_age(row[11])
            x['state'] = row[12]
            x['gender'] = get_gender(row[13])
            x['job'] = row[14]
            status = int(row[6])
            price = int(row[15])
            n_samples += 1
            data['rant'].append(unicodedata.normalize('NFKC', row[5]))
            data['userprofile'].append(x)
            target.append(target_func(status, price))
        logging.info('Finished loading data. (read: {} errors: {})'.format(n_samples, parse_errors))
    return Bunch(data=data,
                 target=target,
                 DESCR="Fuman DB csv dump dataset")


def load_fuman_rant(file_path, target_func=fuman_gvb_target):
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
            if len(row) is 16:
                price = int(row[15])
            else:
                price = 0
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
