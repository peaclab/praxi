from multiprocessing import Lock

import logging
import logging.config

import os
from os import listdir
from os.path import isfile, join

from pathlib import Path
import random
import time
import yaml
import pickle
import copy
import argparse

from sklearn.base import BaseEstimator
from tqdm import tqdm

import numpy as np
from numpy import savetxt
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from hybrid_mod import Hybrid

CHANGESET_ROOT = Path('~/praxi/caches/changesets/').expanduser()

def get_free_filename(stub, directory, suffix=''):
    counter = 0
    while True:
        file_candidate = '{}/{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            print("file exists")
            counter += 1
        else:  # No match found
            print("no file")
            if suffix=='.p':
                print("will create pickle file")
            elif suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate


def parse_ts(tagset_names, ts_dir):
    """ Function for parsing a list of tagsets
    input: - tagset_names: a list of names of tagsets
           - ts_dir: the directory in which they are located
    output: - tags: list of lists-- tags for each tagset name
            - labels: application name corresponding to each tagset
    """
    tags = []
    labels = []
    for ts_name in tqdm(tagset_names):
            ts_path = ts_dir + '/' + ts_name
            with open(ts_path, 'r') as stream:
                tagset = yaml.load(stream)
            if 'labels' in tagset:
                # Multilabel changeset
                labels.append(tagset['labels'])
            else:
                labels.append(tagset['label'])
            tags.append(tagset['tags'])
    return tags, labels

def initial_train():
    # get result file name
    train_init = '/home/ubuntu/praxi/it_tagsets/sl_train'
    #train_it = '/home/ubuntu/praxi/it_tagsets/it_train'
    test = '/home/ubuntu/praxi/it_tagsets/first_test'
    outdir = '/home/ubuntu/praxi/results/iterative'

    vwargs = '-b 26 --learning_rate 1.5 --passes 10'
    resfile_name = get_free_filename('iterative-hybrid', outdir, suffix='.yaml')
    suffix = 'initial'
    iterative = True
    modfile='initial_model.vw'

    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args= vwargs, suffix=suffix, iterative=iterative,
                 use_temp_files=False, vw_modelfile=modfile)
    # GET TAGSETS
    ### THIS IS NOT DONE!!!
    train_init_names = [f for f in listdir(train_init) if (isfile(join(train_init, f))and f[-3:]=='tag')]
    #train_it_names = [f for f in listdir(train_it) if (isfile(join(train_it, f)) and f[-3:]=='tag')]
    test_names = [f for f in listdir(test) if (isfile(join(test, f)) and f[-3:]=='tag')]

    train_init_tags, train_init_labels = parse_ts(train_init_names, train_init)
    test_tags, test_labels = parse_ts(test_names, test)

    # Now train iteratively!
    get_scores_new(clf, resfile_name, train_init_tags, train_init_labels, test_tags, test_labels)


def new_train():
    train_it = '/home/ubuntu/praxi/it_tagsets/it_train'
    test = '/home/ubuntu/praxi/it_tagsets/first_test'
    outdir = '/home/ubuntu/praxi/results/iterative'

    old_model_name = 'initial_model.vw'
    idx_file = '/home/ubuntu/praxi/idx_file.yaml'

    vwargs = '-b 26 --learning_rate 1.5 --passes 10'

    modfile = 'new_model.vw'

    resfile_name = get_free_filename('iterative-hybrid-newdata', outdir, suffix='.yaml')
    suffix = 'iteration'
    iterative = True

    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args= vwargs, suffix=suffix, iterative=iterative,
                 use_temp_files=False, vw_modelfile=modfile)

    train_it_names = [f for f in listdir(train_it) if (isfile(join(train_it, f)) and f[-3:]=='tag')]
    test_names = [f for f in listdir(test) if (isfile(join(test, f)) and f[-3:]=='tag')]

    train_it_tags, train_it_labels = parse_ts(train_it_names, train_it)
    test_tags, test_labels = parse_ts(test_names, test)

    with open(idx_file, 'r') as stream:
        idx_dic = yaml.load(stream)

    get_scores_new(clf, resfile_name, train_it_tags, train_it_labels,
               test_tags, test_labels, existing_model=old_model_name, old_idx=idx_dic)



def get_scores_new(clf, res_name, X_train, y_train, X_test, y_test,
               existing_model=None, old_idx=None, binarize=False):
    #Gets two lists of changeset ids, does training+testing
    clf.fit(X_train, y_train, existing_model=existing_model, idx_dic=old_idx)
    print("Training complete")
    input("Press Enter to continue...")
    preds = clf.predict(X_test)

    # create a dictionary containing true labels, predictions, missed, and # missed
    missed_apps = []
    num_missed = 0
    for label, pred in zip(y_test, preds):
        if label != pred:
            # create a tuple
            wrong_pred = (label, pred)
            missed_apps.append(wrong_pred)
            num_missed += 1
    res_dict = {'missed apps': missed_apps, 'number missed': num_missed}
    with open(res_name, 'w') as outfile:
        yaml.dump(res_dict, outfile, default_flow_style=False)


if __name__ == '__main__':
    initial_train()
    #input("Press Enter to continue...")
    #new_train()






##########################

"""
def initial_train():
    # get result file name
    train_init = '/home/ubuntu/praxi/it_tagsets/sl_train'
    #train_it = '/home/ubuntu/praxi/it_tagsets/it_train'
    test = '/home/ubuntu/praxi/it_tagsets/first_test'
    outdir = '/home/ubuntu/praxi/results/iterative'

    vwargs = '-b 26 --learning_rate 1.5 --passes 10'
    resfile_name = get_free_filename('iterative-hybrid', outdir, suffix='.yaml')
    suffix = 'initial'
    iterative = True
    modfile='initial_model.vw'

    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args= vwargs, suffix=suffix, iterative=iterative,
                 use_temp_files=False, vw_modelfile=modfile)
    # GET TAGSETS
    ### THIS IS NOT DONE!!!
    train_init_names = [f for f in listdir(train_init) if (isfile(join(train_init, f))and f[-3:]=='tag')]
    #train_it_names = [f for f in listdir(train_it) if (isfile(join(train_it, f)) and f[-3:]=='tag')]
    test_names = [f for f in listdir(test) if (isfile(join(test, f)) and f[-3:]=='tag')]

    train_init_tags, train_init_labels = parse_ts(train_init_names, train_init)
    test_tags, test_labels = parse_ts(test_names, test)

    # Now train iteratively!
    get_scores_new(clf, resfile_name, train_init_tags, train_init_labels, test_tags, test_labels)

    train_it = '/home/ubuntu/praxi/it_tagsets/it_train'
    old_model_name = 'initial_model.vw'
    test = '/home/ubuntu/praxi/it_tagsets/first_test'

    modfile = 'new_model.vw'

    resfile_name2 = get_free_filename('iterative-hybrid-newdata', outdir, suffix='.yaml')

    train_it_names = [f for f in listdir(train_it) if (isfile(join(train_it, f)) and f[-3:]=='tag')]

    train_it_tags, train_it_labels = parse_ts(train_it_names, train_it)

    print("Adding new data!")
    get_scores_new(clf, resfile_name2, train_it_tags, train_it_labels,
               test_tags, test_labels)



def get_scores_new(clf, res_name, X_train, y_train, X_test, y_test,
               existing_model=None, binarize=False):
    # Gets two lists of changeset ids, does training+testing
    clf.fit(X_train, y_train, existing_model=existing_model)
    print("Training complete")
    input("Press Enter to continue...")
    preds = clf.predict(X_test)

    # create a dictionary containing true labels, predictions, missed, and # missed
    missed_apps = []
    num_missed = 0
    for label, pred in zip(y_test, preds):
        if label != pred:
            # create a tuple
            wrong_pred = (label, pred)
            missed_apps.append(wrong_pred)
            num_missed += 1
    res_dict = {'missed apps': missed_apps, 'number missed': num_missed}
    with open(res_name, 'w') as outfile:
        yaml.dump(res_dict, outfile, default_flow_style=False)
"""
