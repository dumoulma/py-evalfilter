import logging
import csv

import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix
import numpy as np

FUMAN_PATH = "../../data/20150930/good-vs-bad-100.csv"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_fuman(filepath=FUMAN_PATH, sparse=True):
    fuman_data = FumanDataset(filepath)
    X, y = zip(*[(r[:-1], r[-1]) for r in fuman_data])
    logging.info("Finished reading data")
    X, y = X, list(y)
    le1 = pp.LabelEncoder()
    encoded_gender = le1.fit_transform([x[23] for x in X])
    le2 = pp.LabelEncoder()
    encoded_state = le2.fit_transform([x[23] for x in X])
    le3 = pp.LabelEncoder()
    encoded_job = le3.fit_transform([x[23] for x in X])
    for x, eg, es, ej in zip(X, encoded_gender, encoded_state, encoded_job):
        x[23] = eg
        x[25] = es
        x[26] = ej
    if sparse:
        X = csr_matrix(list(X), dtype=np.float16)
    return X, y


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
