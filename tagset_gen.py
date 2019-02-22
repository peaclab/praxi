# Sadie Allen
# 2/20/2019
# Script function:
#      - given a changeset, create a file containing the tagset
#      - also carry over any other relevant information, e.g. if
#        the changeset is multilabel, specify number of files downloaded
#        within the time

# (This file does not have to work from command line, all its functions will be called from main file)

# Imports
from collections import Counter
import logging
import logging.config
from hashlib import md5
from multiprocessing import Lock
import os
from pathlib import Path
import random
import tempfile
import time
import yaml

import envoy
from joblib import Memory
from sklearn.base import BaseEstimator
from tqdm import tqdm

from columbus.columbus import columbus
from columbus.columbus import refresh_columbus

#  Change for use on local machine
LOCK = Lock()
CHANGESET_ROOT = Path('~/caches/changesets/').expanduser()
COLUMBUS_CACHE = Path('/home/ubuntu/caches/columbus-cache-2').expanduser()
memory = Memory(cachedir='/home/ubuntu/caches/joblib-cache', verbose=0)

def get_changeset(cs_fname, iterative=False):
    changeset = None
    for csfile in CHANGESET_ROOT.glob(cs_fname):
        if changeset is not None:
            raise IOError(
                "Too many changesets match the file name")
        #with open(csfile, 'r') as f:
        with csfile.open('r') as f:
            changeset = yaml.load(f)
    if changeset is None:
        raise IOError("No changesets match the csid {}".format(csid))
    if 'changes' not in changeset or (
            'label' not in changeset and 'labels' not in changeset):
        #logging.error("Malformed changeset, id: %d, changeset: %s",
        #              csid, csfile)
        raise IOError("Couldn't read changeset")
    #print(changeset)
    return changeset


def _get_tags(self, X):
    #logging.info("Getting tags for input set %s" % len(X))
    #if self.pass_files_to_vw:
    #    return _get_filename_frequencies(X, disable_tqdm=(not self.tqdm),
    #                                     freq_threshold=self.freq_threshold)
    return _get_columbus_tags(X, disable_tqdm=(not self.tqdm),
                              freq_threshold=self.freq_threshold,
                              return_freq=self.pass_freq_to_vw)

class Columbus(BaseEstimator):
    """ scikit style class for columbus """
    def __init__(self, freq_threshold=2, tqdm=True):
        """ Initializer for columbus. Do not use multiple instances
        simultaneously.
        """
        self.freq_threshold = freq_threshold
        self.tqdm = tqdm

    def fit(self, X, y):
        pass

    def predict(self, X):
        tags = self._columbize(X)
        result = []
        for tagset in tags:
            result.append(max(tagset.keys(), key=lambda key: tagset[key]))
        return result

    def _columbize(self, X):
        mytags =  _get_columbus_tags(X, disable_tqdm=(not self.tqdm),
                                     freq_threshold=self.freq_threshold,
                                     return_freq=True)
        result = []
        for tagset in mytags:
            tagdict = {}
            for x in tagset:
                key, value = x.split(':')
                tagdict[key] = value
            result.append(tagdict)
        return result

@memory.cache
def _get_filename_frequencies(X, disable_tqdm=False, freq_threshold=2):
    #logging.info("Getting filename frequencies for %d changesets", len(X))
    tags = []
    for changeset in tqdm(X, disable=disable_tqdm):
        c = Counter()
        for filename in changeset:
            c.update(filename.split(' ')[1].split('/'))
        del c['']
        tags.append(['{}:{}'.format(tag.replace(':', '').replace('|', ''), freq)
                     for tag, freq in dict(c).items() if freq > freq_threshold])
    return tags

def _get_columbus_tags(X, disable_tqdm=False,
                       return_freq=True,
                       freq_threshold=2):
    print('Getting columbus output for %d changesets', len(X))
    tags = []
    for changeset in tqdm(X, disable=disable_tqdm):
        tag_dict = columbus(changeset, freq_threshold=freq_threshold)
        print(tag_dict)
        if return_freq:
            tags.append(['{}:{}'.format(tag, freq) for tag, freq
                         in tag_dict.items()])
        else:
            tags.append([tag for tag, freq in tag_dict.items()])
    return tags


if __name__ == '__main__':
    # Test: generate a tagset for a single changeset

    changeset_names = ['php.22689.rp.ubx.ts.yaml','rrdtool.24421.rp.ubx.ts.yaml','mksh.106994.yaml']
    changesets = [get_changeset(i) for i in changeset_names]
    print("Number of changesets:", len(changesets))
    tags = _get_filename_frequencies(changesets)
    print(tags)
