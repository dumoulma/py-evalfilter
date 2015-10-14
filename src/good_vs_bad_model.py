#!/usr/bin/env python
import logging

import click
from datasets.fuman import load_fuman
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


@click.command()
@click.argument('input_data', type=click.Path(), nargs=1)
def main(input_data):
    X, y = load_fuman(input_data[0])
    clf = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=1, random_state=0, n_jobs=8)
    scores = cross_val_score(clf, X, y, n_jobs=8)
    print("scores:", scores)


if __name__ == "__main__":
    main()
