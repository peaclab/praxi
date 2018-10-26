import sys
import os
import optparse
import pickle
import logging
import glob
import yaml
from datetime import datetime

from .trie import Trie
from .tags import filtertags

FILTER_PATH_TOKENS = ['usr', 'bin', 'proc', 'sys', 'etc', 'local', 'src',
                      'dev', 'home', 'root', 'lib', 'pkg', 'sbin', 'share',
                      'cache']

COLUMBUS_CACHE = {}


def columbus(changeset, freq_threshold=2):
    """ Get labels from single changeset """
    key = str(sorted(changeset))
    if key not in COLUMBUS_CACHE:
        COLUMBUS_CACHE[key] = run_file_paths_discovery2(
            filtertags, changeset, freq_threshold=freq_threshold)
    return COLUMBUS_CACHE[key]


def run_file_paths_discovery2(filtertags, changeset, freq_threshold=2):
    ftrie = Trie(frequency_limit=freq_threshold)
    for filepath in changeset:
        pathtokens = filepath.split('/')
        for token in pathtokens:
            if token not in FILTER_PATH_TOKENS:
                ftrie.insert(token)

    softtags = ftrie.get_all_tags()
    for tag in filtertags:
        softtags.pop(tag, None)
    return softtags
