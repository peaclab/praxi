#!/usr/bin/env python3

import os
import pickle
import yaml

import click

from main import print_multilabel_results

@click.command()
@click.option('--result', type=click.File('r'),
              help='Output of vw')
@click.option('--table', type=click.File('r'),
              help='Table of labels vs. numbers')
@click.option('--truth', type=click.File('r'),
              help='Actual labels')
@click.option('--output', type=click.Path(dir_okay=True),
              help='Folder to output')
def score_vw(result, table, truth, output):
    """ Returns the scores for vw """
    label_dict = yaml.load(table)
    resfile = 'tmp_results.pkl'
    y_test = []
    ntags = []
    for line in truth:
        labels = [x.strip() for x in line.split(' ')]
        y_test.append(labels)
        ntags.append(len(labels))

    preds = []
    for line, ntag in zip(result, ntags):
        probas = {}
        for word in line.split(' '):
            label, proba = word.split(':')
            probas[label] = proba
        top_k = []
        for i in range(ntag):
            tag = min(probas.keys(), key=lambda key: probas[key])
            probas.pop(tag)
            top_k.append(label_dict[int(tag)])
        preds.append(top_k)
    assert len(preds) == len(y_test)

    with open(resfile, 'wb') as f:
        pickle.dump([(y_test, preds)], f)
    print_multilabel_results(resfile, output)
    os.unlink(resfile)

if __name__ == '__main__':
    score_vw()
