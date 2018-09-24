import sys
import os
import optparse
import pickle
import logging
import glob
import yaml
from datetime import datetime

from .pytrie.trie import Trie

FILTER_PATH_TOKENS = ['usr', 'bin', 'proc', 'sys', 'etc', 'local', 'src',
                      'dev', 'home', 'root', 'lib', 'pkg', 'sbin', 'share',
                      'cache']


def columbus(changeset,
             systagfile='/home/centos/hybrid-method/columbus/systags/ubuntu-1404'):
    """ Get labels from single changeset """
    with open(systagfile, 'rb') as sysfp:
        systags = pickle.load(sysfp)

    result = run_file_paths_discovery2(systags['paths'], changeset)
    return {tag.replace(':', '').replace('|', ''): freq
            for tag, freq in result.items()}


def run_file_paths_discovery2(filtertags, changeset):
    ftrie = Trie()
    for filepath in changeset:
        pathtokens = filepath.split('/')
        for token in pathtokens:
            if token != '' and token not in FILTER_PATH_TOKENS:
                ftrie.insert(token)

    softtags = {}
    res = ftrie.get_all_tags()
    for tag in res:
        if tag in filtertags:
            continue
        softtags[tag] = res[tag]
    return softtags
