#!/usr/bin/env python3

import pickle
from pathlib import Path
import yaml

from tqdm import tqdm

PROJECT_ROOT = Path('~/hybrid-method').expanduser()
CHANGESET_ROOT = Path('~/yaml/testing/').expanduser()


def get_changeset(csid):
    for idx, csfile in enumerate(CHANGESET_ROOT.glob('*{}*'.format(csid))):
        if idx > 0:
            raise IOError("Too many changesets match the csid {}".format(csid))
        with csfile.open('r') as f:
            changeset = yaml.load(f)
    return changeset


def get_scores(test_set, train_set):
    """ Gets two lists of changeset ids """
    print(len(test_set), len(train_set))


def main():
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


if __name__ == '__main__':
    main()
