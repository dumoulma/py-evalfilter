import logging
import csv
from datetime import date

import unicodedata
import datasets.features as cf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

GENDER = {0: "unk", 1: 'male', 2: 'female'}
BOOLEAN = {0: 'False', 1: 'True'}


def manual_features_header():
    header = "hasIndustry,hasOccupation,hasCompany,hasProductName,hasProposal,empathies,"
    header += "birthyear,state,gender,job,"
    header += cf.get_header()
    return header


def load_rants(filepath):
    for row in FumanDataset(filepath):
        if len(row) is not 16:
            logging.debug("Badly formated row: {}".format(row))
            continue
        # must replicate the behaviour of load_fuman_csv
        try:
            int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[6]), int(row[7]), int(row[8]), int(row[15])
        except ValueError as ve:
            logging.warning("Parse problem for rant {} ({})".format(row[0], ve))
            continue
        yield unicodedata.normalize('NFKC', row[5])


def load_target_rants(filepath, target, target_var_func):
    for row in FumanDataset(filepath):
        if len(row) is not 16:
            logging.debug("Badly formated row: {}".format(row))
            continue
        try:
            int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[6]), int(row[7]), int(row[8]), int(row[15])
            price = int(row[15])
            status = int(row[6])
        except ValueError as ve:
            logging.warning("Parse problem for rant {} ({})".format(row[0], ve))
            continue
        if target_var_func(status, price) is not target:
            continue
        yield unicodedata.normalize('NFKC', row[5])


def load_fuman_csv(filepath, target_var_func=None):
    """
    Loads the fuman data from the dump CSV file, with all features ready for use by an estimator.
    This is a generator that yields instances one by one.

    :param filepath: path to CSV input file
    :param target_var_func: Callable that produces the target variable from combination of price and status
    :return: list of variables for the next instance
    """
    for i, row in enumerate(FumanDataset(filepath)):
        if len(row) is not 16:
            logging.debug("Badly formated row: expected 16 fields, got {}".format(len(row)))
            continue
        x = list()
        try:
            x.append(to_binary_categorical(row[1]))  # has industry
            x.append(to_binary_categorical(row[2]))  # has occupation
            x.append(to_binary_categorical(row[3]))  # has company
            x.append(to_binary_categorical(row[4]))  # has product name
            x.append(to_binary_categorical(row[7]))  # has proposal
            x.append(int(row[8]))  # empathies
            x.append(get_age(row[11]))  # birth year
            x.append(row[12])  # state
            x.append(get_gender(row[13]))  # gender
            x.append(row[14])  # job
            x = cf.rant_text_features(x, rant=unicodedata.normalize('NFKC', row[5]))
            status = int(row[6])
            price = int(row[15])
        except ValueError as ve:
            logging.warning("Parse problem for rant id:{} ({})".format(row[0], ve))
            continue
        # target should be added last
        if target_var_func is None:
            x.append(status)  # status
            x.append(price)  # price
        else:
            x.append(target_var_func(status, price))  # target
        if i % 1000 is 0:
            logging.debug("{}: {}".format(i, x))
        yield x


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


class FumanDataset(object):
    """
        Iterable of the raw CSV data from the dump of the FumanDB

        :returns row from CSV as a String
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar="'")
            self.headers = next(reader)  # skip headers
            logging.debug("Got headers: {}".format(self.headers))
            for row in reader:
                yield row


if __name__ == "__main__":
    pass
    # s = 'ミッドソールにはSpEVAと、Solyteを組み合わせたフルイドライドを採用し、流れるようになめらかな走り心地が実現。'
    # tokens = _tokenize(s)
    #
    # # id,hasIndustry,hasOccupation,hasCompany,hasProductName,rants,status,hasProposals,empathies,hasLatitude,
    # # hasLongitude,birth_year,state,gender,job,price
