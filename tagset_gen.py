#!/usr/bin/python3

# 2/20/2019
# Script function:
#      - given a directory of changesets, create a directory containing the tagsets
#      - also carry over any other relevant information, e.g. if
#        the changeset is multilabel, specify number of files downloaded
#        within the time

# Imports
from collections import Counter
#import logging
#import logging.config
#from hashlib import md5
from multiprocessing import Lock

import os
from os import listdir
from os.path import isfile, join
import sys


from pathlib import Path
import random
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


@memory.cache # what does this do...?
def parse_cs(changeset_names, multilabel=False, iterative=False):
    # input: list of changeset names (strings)
    # output: a list of labels and a list of features (list of filepaths of changed files)
    features = []
    labels = []
    for cs_name in tqdm(changeset_names):
            changeset = get_changeset(cs_name, iterative=iterative)
            if multilabel:
                if 'labels' in changeset:
                    labels.append(changeset['labels'])
                else:
                    labels.append(changeset['label'])
            else:
                labels.append(changeset['label'])
            features.append(changeset['changes'])
    return features, labels

def get_changeset(cs_fname, iterative=False):
    # input: file name of a *single* changeset
    # output:
    changeset = None
    #print(CHANGESET_ROOT.glob(cs_fname))
    print(cs_fname)
    for csfile in CHANGESET_ROOT.glob(cs_fname):
        if changeset is not None:
            raise IOError(
                "Too many changesets match the file name")
        #with open(csfile, 'r') as f:
        with csfile.open('r') as f:
            changeset = yaml.load(f)
    if changeset is None:
        raise IOError("No changesets match the name")
    if 'changes' not in changeset or (
            'label' not in changeset and 'labels' not in changeset):
        #logging.error("Malformed changeset, id: %d, changeset: %s",
        #              csid, csfile)
        raise IOError("Couldn't read changeset")
    #print(type(changeset))
    return changeset


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
    # Returns a list of tags and their frequency (as strings)
    tags = []
    for changeset in tqdm(X, disable=disable_tqdm):
        tag_dict = columbus(changeset, freq_threshold=freq_threshold)
        if return_freq:
            tags.append(['{}:{}'.format(tag, freq) for tag, freq
                         in tag_dict.items()])
        else:
            tags.append([tag for tag, freq in tag_dict.items()])
    return tags

def create_tagset_names(changeset_names):
    # input: list of changeset names
    # output: list of names for tagsets created for these changesets
    tagset_names = []
    for name in changeset_names:
        new_tagname = name[:-4] + "tag"
        #print(new_tagname)
        tagset_names.append(new_tagname)
    print('TAGSET_NAMES ELEM', tagset_names[0])
    return tagset_names

def get_changeset_names(cs_dir):
    # takes a directory name and returns all changeset files within the directory
    all_files = [f for f in listdir(cs_dir) if isfile(join(cs_dir, f))]
    changeset_names = [f for f in all_files if ".yaml" in f and ".tag" not in f]
    return changeset_names

def get_ids(changeset_names):
    # return the ids of all changesets in a list (between first and second '.')
    c_ids = []
    c = '.'
    for cs_name in changeset_names:
        idxs = [pos for pos, char in enumerate(cs_name) if char == c]
        curr_id = cs_name[idxs[0]+1:idxs[1]]
        c_ids.append(curr_id)
    return c_ids

if __name__ == '__main__':
    # Test: generate tagsets for a small list of tagsets
    # COMMAND LINE ARGS
    #cs_dir = '/home/ubuntu/praxi/week5/changeset_sub'
    #ts_dir = '/home/ubuntu/praxi/week5/tagset_files'

    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))

    """
    changeset_names = get_changeset_names(cs_dir)
    tagset_names = create_tagset_names(changeset_names) # names for new tagset files!!!
    ids = get_ids(changeset_names)
    #print(tagset_names)
    #changesets = []
    #features = []

    changesets, labels = parse_cs(changeset_names)
    print("Number of changesets:", len(changesets))
    tags = _get_columbus_tags(changesets)

    # add tags, labels, ids to yaml files
    for i in range(len(tagset_names)):
        cur_dict = {'labels': labels[i], 'id' : ids[i], 'tags': tags[i]}
        print(tagset_names[i])
        cur_fname = ts_dir + '/' + tagset_names[i]
        with open(cur_fname, 'w') as outfile:
            yaml.dump(cur_dict, outfile, default_flow_style=False)"""
