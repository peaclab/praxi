import os
from os import listdir
from os.path import isfile, join

from pathlib import Path
import random
#import tempfile
import time
import yaml
#import pickle
#import copy
import argparse
from tqdm import tqdm

import re

def find_repeats(tags, freqs):
    """Goes through list of tags (and frequencies) and finds those that repeat
       Deletes repetitions and adds frequencies
    """
    new_tags = []
    new_freqs = []
    for tag, freq in zip(tags, freqs):
        if tag not in new_tags:
            new_tags.append(tag)
            new_freqs.append(freq)
        else:
            # find index of previous occurence
            old_idx = new_tags.index(tag)
            new_freqs[old_idx] = str(int(freq) + int(new_freqs[old_idx]))
    return new_tags, new_freqs

def simple_tag(tag):
    """ Given a single tag, remove punctuations (or punctuation and numbers?)"""
    ntag = tag
    if tag[-1] in '[!@#$._-]':
        ntag = ntag[:-1]
    if tag[0] in '[!@#$._-]':
        ntag = ntag[1:]
    return ntag

def tag_comb(tag_list):
    """ Given a list of tags (with frequencies), combine near-identical tags and
        add their frequencies"""
    # First, separate frequencies from tags and create two lists, get rid of
    # extra punctuation
    tags = []
    freqs = []
    for tagstr in tag_list:
        tag, freq = tagstr.split(':')
        ntag = simple_tag(tag)
        tags.append(ntag)
        freqs.append(freq)
    n_tags, n_freqs = find_repeats(tags,freqs)
    return n_tags, n_freqs
    #print(tags, freqs)

def create_file(fname, tags, freqs, labels, id):
    t_w_freqs = []
    #if is_instance(labels[i], list)):
    for freq, tag in zip(freqs, tags):
        t_w_freqs.append(tag + ':' + freq)
    cur_dict = {'labels': labels, 'id':id, 'tags':t_w_freqs}
    with open(fname, 'w') as outfile:
        yaml.dump(cur_dict, outfile, default_flow_style=False)

# only for single labels right now
def edit_tagsets(ts_names, og_path, new_path):
    # Load in yaml file
    for name in tqdm(ts_names):
        curpath = og_path + '/' + name
        with open(curpath) as stream:
            data = yaml.load(stream)
        tag_list = data['tags']
        label = data['label']
        id = data['id']
        tags, freqs = tag_comb(tag_list)
        newfilename = new_path + '/' + name
        create_file(newfilename, tags, freqs, label, id)

if __name__ == '__main__':
    # Pass in a starting directory
    work_dir = os.path.abspath('.')

    parser = argparse.ArgumentParser(description='Arguments for tagset generation.')
    parser.add_argument('-t','--tagdir', help='Path to original directory.', required=True)
    parser.add_argument('-n', '--newdir', help='Path to new tagset directory.', default=None)

    args = vars(parser.parse_args())

    og_path = args['tagdir']
    new_path = args['newdir']

    ts_names = [f for f in listdir(og_path) if (isfile(join(og_path, f))and f[-3:]=='tag')]
    print(len(ts_names))
    # can name them the same thing... w .e extension?

    edit_tagsets(ts_names, og_path, new_path)
