#!/usr/bin/env python3

import logging
import logging.config
import os
import sys
import pickle
from pathlib import Path
import tempfile
import yaml
from time import sleep

import envoy
from tqdm import tqdm

from columbus.columbus import columbus

PROJECT_ROOT = Path('~/hybrid-method').expanduser()
CHANGESET_ROOT = Path('~/yaml/testing/').expanduser()


def main():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'standard': {
                'format': '%(asctime)s %(levelname)-7s %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    })
    get_scores([101289, 102580, 102585, 99234], [101289, 102580, 102585, 99234])
    with (PROJECT_ROOT / 'changeset_sets' /
          'threek_dirty_chunks.p').open('rb') as f:
        threeks = pickle.load(f)
    with (PROJECT_ROOT / 'changeset_sets' /
          'tenk_clean_chunks.p').open('rb') as f:
        tenks = pickle.load(f)
    for idx, test_set in enumerate(threeks):
        train_idx = [0, 1, 2]
        train_idx.remove(idx)
        train_set = threeks[train_idx[0]] + threeks[train_idx[1]]
        get_scores(test_set, train_set)
        for idx, extra_cleans in enumerate(tenks):
            train_set += extra_cleans
            get_scores(test_set, train_set)


class Hybrid:
    """ scikit style class for hybrid method """
    def __init__(self, k=15, vw_binary='/home/ubuntu/bin/vw',
                 vw_args='-c --loss_function hinge --redefine :=ctags -q cc '
                         '-b 25 --passes 50 --oaa 78 -l 0.4'):
        self.k = k
        self.vw_args = vw_args
        self.vw_binary = vw_binary
        self.indexed_labels = {}
        # TODO: delete this file at __del__? after debugging
        self.vw_modelfile = tempfile.NamedTemporaryFile('w', delete=False)
        logging.info('Started hybrid model, vw_modelfile: %s',
                     self.vw_modelfile.name)

    def fit(self, X, y):
        logging.info('Training started')
        counter = 1
        for label in set(y):
            self.indexed_labels[label] = counter
            counter += 1
        tags = self._columbize(X)
        f = tempfile.NamedTemporaryFile('w', delete=False)
        for tag, label in zip(tags, y):
            f.write('{} 1.0 {} | {}\n'.format(
                self.indexed_labels[label],
                label, ' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting training', f.name)
        c = envoy.run(
            '{vw_binary} {vw_input} {vw_args} -f {vw_modelfile}'.format(
                vw_binary=self.vw_binary, vw_input=f.name,
                vw_args=self.vw_args, vw_modelfile=self.vw_modelfile.name)
        )
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. out: %s, err: %s',
                c.std_out, c.std_err)
        os.unlink(f.name)

        # # For testing
        # cd /home/ubuntu/vw/results
        # vw -t -i ../chosen_dirty.model ../rp_ts_vw_new.in -p rp_ts_vw_new.out.11
        # python2.7 score.py rp_ts_vw_new.out.11

    def predict(self, X):
        tags = self._columbize(X)
        f = tempfile.NamedTemporaryFile('w', delete=False)
        for tag in tags:
            f.write('| {}\n'.format(' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting testing', f.name)
        c = envoy.run(
            '{vw_binary} {vw_input} -p /dev/stdout -i {vw_modelfile}'.format(
                vw_binary=self.vw_binary, vw_input=f.name,
                vw_modelfile=self.vw_modelfile.name)
        )
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. out: %s, err: %s',
                c.std_out, c.std_err)
        os.unlink(f.name)
        return c.std_out.split()

    def _columbize(self, X):
        logging.info('Getting columbus output for %d changesets', len(X))
        tags = []
        for changeset in X:
            tags.append(columbus(changeset, k=self.k))
        return tags

    def score(self, X, y):
        predictions = self.predict(X)


def get_changeset(csid):
    changeset = None
    for csfile in CHANGESET_ROOT.glob('*{}*'.format(csid)):
        if changeset is not None:
            raise IOError("Too many changesets match the csid {}".format(csid))
        with csfile.open('r') as f:
            changeset = yaml.load(f)
    if changeset is None:
        raise IOError("No changesets match the csid {}".format(csid))
    return changeset


def parse_csids(csids):
    """ Returns labels and features from csids, features are file sets
    file sets: list of string of format '644 /usr/.../file' """
    features = []
    labels = []
    for csid in csids:
        changeset = get_changeset(csid)
        labels.append(changeset['label'])
        features.append(changeset['changes'])
    return features, labels


def get_scores(test_set, train_set):
    """ Gets two lists of changeset ids, does training+testing """
    clf = Hybrid()
    X, y = parse_csids(train_set)
    clf.fit(X, y)
    X, y = parse_csids(test_set)
    clf.score(X, y)


if __name__ == '__main__':
    main()
