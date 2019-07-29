#!/usr/bin/env python3
""" Script function:
    - two modes:
        1) Cross Validation: takes a single directory of tagsets and runs the
                Praxi application discovery algorithm, dividing the
                data into folds and repeatedly running the experiment
                with each fold being the test set
        2) "Real World" Experiment: takes two directories of tagsets, one for
                training, one for testing, and runs Praxi once, first training
                the model and then evaluating its accuracy using the test
                directory
    - inputs/arguments:
        * -t [directory name]: path to training tagset directory (REQUIRED)
        * -s [directory name]: path to testing tagset directory (only required
                               for experiment 2)
        * -o [directory name]: path to desired result directory
        * -m: run a multilabel experiment
        * -w [args]: customize arguments for VW learning algorithm
        * -n [# of folds]: number of folds to use if running an experiment with
                           cross validation
        * -f: output the full results instead of a summary
        * -v: increase verbosity of log messages
    - output: outputs a text file containing statistics about the performance
              of the algorithm; choice between summary or full result file
"""

# Imports
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

from hybrid_tags import Hybrid

LOCK = Lock()

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


def get_free_filename(stub, directory, suffix=''):
    """ Get a file name that is unique in the given directory
    input: the "stub" (string you would like to be the beginning of the file
        name), the name of the directory, and the suffix (denoting the file type)
    output: file name using the stub and suffix that is currently unused
        in the given directory
    """
    counter = 0
    while True:
        file_candidate = '{}/{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            logging.info("file exists matching the string %s", file_candidate)
            counter += 1
        else:  # No match found
            logging.info("no file exists matching the string %s", file_candidate)
            if suffix=='.p':
                logging.info("will create pickle file")
            elif suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate

def fold_partitioning(ts_names, n=3):
    """ Partitions tagsets into folds for cross validation, making sure to avoid
    class imbalance by putting an (approximately) equal number of examples of
    each application in every fold
    input: list of tagset names and the desired number of folds
    output: list containing n lists of balanced tagsets
    """
    folds = [[] for _ in range(n)]

    just_progs = []
    for name in ts_names:
        ts_name_comps = name.split('.')
        just_progs.append(ts_name_comps[0])

    prog_set = set(just_progs)
    unique_progs = (list(prog_set))

    prog_partition = [[] for _ in range(len(unique_progs))]

    for name in ts_names:
        name_comps = name.split('.')
        just_pname = name_comps[0]
        for i, prog in enumerate(unique_progs):
            if just_pname == prog:
                prog_partition[i].append(name)

    for ts_names in prog_partition:
        for idx, name in enumerate(ts_names):
            folds[idx % n].append(name)
    return folds

def single_label_experiment(nfolds, names, name, tagset_directory, resfile_name, outdir, vwargs):
    """ Run a single-label experiment (with or without cross validation)
    input: number of folds, training directory path, name of result file,
            result directory path, Vowpal Wabbit arguments, result type
            (summary or full), testing directory path
    output: pickle file with label predictions and text file with experiment
            performance statistics both written to output directory
    """
    # instantiate hybrid object
    suffix = 'single'
    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args=vwargs, suffix=suffix, iterative=False,
                 use_temp_files=True)
    resfile = open(resfile_name, 'wb')
    results = []
    if(len(names) == 0):
        raise ValueError("No tagsets in provided directory!")
    folds = fold_partitioning(names, n=nfolds)
    for idx, fold in enumerate(folds):
        test_tagset_names = fold
        train_idx_list = list(range(len(folds)))
        train_idx_list.remove(idx)
        logging.info("Training folds: %s", str(train_idx_list))
        train_tagset_names = []
        for i in train_idx_list:
            train_tagset_names += folds[i]
        test_tags, test_labels = parse_ts(test_tagset_names, tagset_directory)
        train_tags, train_labels = parse_ts(train_tagset_names, tagset_directory)
        results.append(get_scores(clf, train_tags, train_labels, test_tags, test_labels))

    pickle.dump(results, resfile)
    # results is a list of tuples
    resfile.close()
    logging.info("Printing results:")
    print_results(resfile_name, outdir, name)

def get_scores(clf, train_tags, train_labels, test_tags, test_labels,
               binarize=False, store_true=False):
    """ Performs training and testing on the given tagsets
    input: model object, tags and labels for training set, tags and labels for
            testing set
    output: list of labels for the test set, list of label predictions given by
            the classifier
    """
    if binarize:
        binarizer = MultiLabelBinarizer()
        clf.fit(train_tags, binarizer.fit_transform(train_labels))
        preds = binarizer.inverse_transform(clf.predict(test_labels))
    else:
        logging.info("Fitting model:")
        clf.fit(train_tags, train_labels) # train model
        logging.info("Generating predictions:")
        preds = clf.predict(test_tags) # predict labels for test set
    return copy.deepcopy(test_labels), preds

def print_results(resfile, outdir, name, result_type='summary', n_strats=1, args=None, iterative=False):
    """ Calculate result statistics and print them to result file
    input: name of result pickle file, path to result directory, type of result
           desired
    output: text file with experiment result statistics
    """
    logging.info("Writing scores to %s", str(outdir))
    with open(resfile, 'rb') as f:
        results = pickle.load(f)
    # # Now do the evaluation!
    # #results = [
    # #    0 => ([x, y, z], <-- true
    # #          [x, y, k]) <-- pred
    # #]
    numfolds = len(results)
    y_true = []
    y_pred = []
    for result in results:
        y_true += result[0]
        y_pred += result[1]

    labels = sorted(set(y_true))
    # these will be all length 1
    classifications = []
    f1_weighted = []
    f1_micro = []
    f1_macro = []
    p_weighted = []
    p_micro = []
    p_macro = []
    r_weighted = []
    r_micro = []
    r_macro = []
    confusions = []
    label_counts = []
    x = y_true
    y = y_pred

    classifications.append(metrics.classification_report(x, y, labels))
    f1_weighted.append(metrics.f1_score(x, y, labels, average='weighted'))
    f1_micro.append(metrics.f1_score(x, y, labels, average='micro'))
    f1_macro.append(metrics.f1_score(x, y, labels, average='macro'))
    p_weighted.append(metrics.precision_score(x, y, labels, average='weighted'))
    p_micro.append(metrics.precision_score(x, y, labels, average='micro'))
    p_macro.append(metrics.precision_score(x, y, labels, average='macro'))
    r_weighted.append(metrics.recall_score(x, y, labels, average='weighted'))
    r_micro.append(metrics.recall_score(x, y, labels, average='micro'))
    r_macro.append(metrics.recall_score(x, y, labels, average='macro'))
    confusions.append(metrics.confusion_matrix(x, y, labels))
    label_counts.append(len(set(x)))

    # this for loop will only run once
    for strat, report, f1w, f1i, f1a, pw, pi, pa, rw, ri, ra, confuse, lc in zip(
            range(n_strats), classifications, f1_weighted, f1_micro, f1_macro,
            p_weighted, p_micro, p_macro, r_weighted, r_micro, r_macro,
            confusions, label_counts):

        if numfolds == 1: # no cross validation
            file_header = (
                "SINGLE LABEL EXPERIMENTAL REPORT:\n" +
                time.strftime("Generated %c\n\n") +
                ('\n Args: {}\n\n'.format(args) if args else '') +
                "EXPERIMENT WITH {} TEST CHANGESETS\n".format(len(y_true)))
            fstub = 'single_exp'
        else:
            file_header = (
                "SINGLE LABEL EXPERIMENTAL REPORT:\n" +
                time.strftime("Generated %c\n\n") +
                ('\n Args: {}\n\n'.format(args) if args else '') +
                "{} FOLD CROSS VALIDATION WITH {} CHANGESETS\n".format(numfolds, len(y_true)))
            fstub = 'single_exp_cv'

        os.makedirs(str(outdir), exist_ok=True) # makes directory if it doesn't exist
        file_header += (
            "F1 SCORE : {:.3f} weighted\n".format(f1w) +
            "PRECISION: {:.3f} weighted\n".format(pw) +
            "RECALL   : {:.3f} weighted\n\n".format(rw))
        if numfolds == 1:
            hits = misses = predictions = 0
            for pred, label in zip(y_true, y_pred):
                if pred == label:
                    hits += 1
                else:
                    misses += 1
                predictions += 1
            str_add = "Preds: " + str(predictions) + "\nHits: " + str(hits) + "\nMisses: " + str(misses)
            file_header += str_add
        fname = get_free_filename(name, outdir, '.txt')
        f = open(fname, "w")
        f.write(file_header) # just need header b/c no confusion matrix
        f.close()

def sep_apps(tag_names):
    app_names = []
    # get all names
    for name in tag_names:
        # do stuff
        pieces = name.split('.')
        a_name = pieces[0]
        if a_name not in app_names:
            app_names.append(a_name)
    sep_list = [ [] for i in range(len(app_names))]

    for name in tag_names:
        pieces = name.split('.')
        a_name = pieces[0]
        sep_list[app_names.index(a_name)].append(name)

    return sep_list

if __name__ == '__main__':
    prog_start = time.time()

    parser = argparse.ArgumentParser(description='Arguments for Praxi software discovery algorithm.')
    parser.add_argument('-t','--traindir', help='Path to training tagset directory.', required=True)
    parser.add_argument('-o', '--outdir', help='Path to desired result directory', default='.')
    # run a single label experiment by default, if --multi flag is added, run a multilabel experiment!
    parser.add_argument('-w','--vwargs', dest='vw_args', default='-b 26 --learning_rate 1.5 --passes 10',
                        help="custom arguments for VW.")
    parser.add_argument('-n', '--nfolds', help='number of folds to use in cross validation', default=1) # make default 1?
    parser.add_argument('-v', '--verbosity', dest='loglevel', action='store_const', const='DEBUG',
                        default='WARNING',help='specify level of detail for log file')
    # IMPLEMENT THIS!

    args = vars(parser.parse_args())

    outdir = os.path.abspath(args['outdir'])
    nfolds = int(args['nfolds'])
    tagset_directory = args['traindir']

    # SET UP LOGGING
    loglevel = args['loglevel']
    stub = 'praxi_exp'
    logfile_name = get_free_filename(stub, outdir, '.log')

    numeric_level = getattr(logging, loglevel, None)
    logging.basicConfig(filename=logfile_name,level=numeric_level)


    vwargs = args['vw_args']
    logging.info("Arguments for Vowpal Wabbit: %s", vwargs)

    tag_names = [f for f in listdir(tagset_directory) if (isfile(join(tagset_directory, f))and f[-3:]=='tag')]

    separated = sep_apps(tag_names)

    for names in separated:
        pieces = names[0].split('.')
        name = pieces[0]
        resfile_name = get_free_filename(name, outdir, '.p') # add arg to set stub?
        single_label_experiment(nfolds, names, name, tagset_directory, resfile_name, outdir, vwargs) # no traim directory

    logging.info("Program runtime: %s", str(time.time()-prog_start))
    print("Program runtime: %s", str(time.time()-prog_start))
