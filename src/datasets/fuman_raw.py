import logging
import csv

import unicodedata
import datasets.features as cf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

GENDER = {0: "unk", 1: 'male', 2: 'female'}


def get_header():
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


def load_fuman_csv(filepath, target_var_func=None) -> list:
    for i, row in enumerate(FumanDataset(filepath)):
        if len(row) is not 16:
            logging.debug("Badly formated row: expected 16 fields, got {}".format(len(row)))
            continue
        x = list()
        try:
            x.append(int(row[1]))  # has industry
            x.append(int(row[2]))  # has occupation
            x.append(int(row[3]))  # has company
            x.append(int(row[4]))  # has product name
            x.append(int(row[7]))  # has proposal
            x.append(int(row[8]))  # empathies
            x.append(row[11])  # birth year
            x.append(row[12])  # state
            x.append(get_gender(row[13]))  # gender
            x.append(row[14])  # job
            x = cf.rant_text_features(x, rant=unicodedata.normalize('NFKC', row[5]))
            status = int(row[6])
            price = int(row[15])
        except ValueError as ve:
            logging.warning("Parse problem for rant {} ({})".format(row[0], ve))
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


def get_gender(field):
    if field == '''\\0''':
        g = 0
    else:
        try:
            g = int(field)
        except ValueError as ve:
            logging.warning("Can't parse gender, set to UNKNOWN (got: {})".format(ve))
            g = 0
    return GENDER[g]


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
            logging.debug("Got headers: {}".format(headers))
            for row in reader:
                yield row


if __name__ == "__main__":
    pass
    # s = 'ミッドソールにはSpEVAと、Solyteを組み合わせたフルイドライドを採用し、流れるようになめらかな走り心地が実現。'
    # tokens = _tokenize(s)
    #
    # # id,hasIndustry,hasOccupation,hasCompany,hasProductName,rants,status,hasProposals,empathies,hasLatitude,
    # # hasLongitude,birth_year,state,gender,job,price
