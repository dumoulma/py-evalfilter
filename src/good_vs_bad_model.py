#!/usr/bin/env python
import logging
import os

import click
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cluster import k_means

from datasets.fuman_base import load_fuman_gvb

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


@click.command()
@click.argument('source', type=click.Path(), nargs=1)
def main(source):
    if os.path.isfile(source):
        raise ValueError("Source must be a directory")

    fuman_dataset = load_fuman_gvb(source)
    X, y = fuman_dataset.data, fuman_dataset.target
    clf = GradientBoostingClassifier()
    scores = cross_val_score(clf, X, y, n_jobs=8)
    print("scores:", scores)


if __name__ == "__main__":
    main()
