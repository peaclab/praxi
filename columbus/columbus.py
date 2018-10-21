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


def columbus(changeset):
    """ Get labels from single changeset """
    return run_file_paths_discovery2(filtertags, changeset)


def run_file_paths_discovery2(filtertags, changeset):
    ftrie = Trie(frequency_limit=2)
    for filepath in changeset:
        pathtokens = filepath.split('/')
        for token in pathtokens:
            if token not in FILTER_PATH_TOKENS:
                ftrie.insert(token)

    softtags = ftrie.get_all_tags()
    for tag in filtertags:
        softtags.pop(tag, None)
    return softtags
