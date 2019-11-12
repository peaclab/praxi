#!/usr/bin/python3
""" Main for new version detection code developed Fall 2019
Sadie Allen
sadiela@bu.edu
"""

""" Inputs
Takes two directories of hybrid tag/changesets: one for training and one for
testing
"""

""" Outputs
Creates a file containing ML performance statistics
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
#import tempfile
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
from basic_rule_class import RuleBasedTags
from collections import OrderedDict
from orderedset import OrderedSet

LOCK = Lock()

#######################################
#   FUNCTIONS for accessing tagsets   #
#######################################
def parse_hyb(hyb_names, data_dir):
    # Arguments: - hyb_names: a list of names of hybrid change/tagsets
    #            - data_dir: the directory in which they are located
    # Returns: - list of dictionaries  containing tags, changes, version label, and app label
    #tags = []
    #labels = []
    hyb_dics = []
    for hyb_name in tqdm(hyb_names):
            data_path = data_dir + '/' + hyb_name
            hyb_dic = get_dic(data_path)
            hyb_dics.append(hyb_dic)
    return hyb_dics


def get_dic(dat_path): # combine with parse_ts
    # Argument: - complete path to a tagset .yaml file
    # Returns:  - tagset dictionary contained in file (tags, labels)
    with open(dat_path, 'r') as stream:
        data_loaded = yaml.load(stream)
    return data_loaded

#######################################
#     MISCELLANEOUS                   #
#######################################

def get_free_filename(stub, directory, suffix=''):
    counter = 0
    while True:
        file_candidate = '{}/{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            #print("file exists")
            counter += 1
        else:  # No match found
            #print("no file")
            if suffix=='.p':
                print("will create pickle file")
            elif suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate

def fold_partitioning(hyb_names, n=3):
    # partition tagsets into folds for cross validation
    folds = [[] for _ in range(n)]

    just_progs = []
    for name in hyb_names:
        hyb_name_comps = name.split('.')
        just_progs.append(hyb_name_comps[0])

    prog_set = set(just_progs)
    unique_progs = (list(prog_set))

    prog_partition = [[] for _ in range(len(unique_progs))]

    for name in hyb_names:
        name_comps = name.split('.')
        just_pname = name_comps[0]
        for i, prog in enumerate(unique_progs):
            if just_pname == prog:
                prog_partition[i].append(name)

    for hyb_names in prog_partition:
        for idx, name in enumerate(hyb_names):
            folds[idx % n].append(name)

    return folds

#################################
#### SINGLE LABEL EXPERIMENT ####
#################################
def single_label_experiment(nfolds, train_path, resfile_name, outdir, vwargs, result_type, test_path=None, print_misses=False):
    # instantiate hybrid object
    suffix = 'single'
    # Create an instance of the Hybrid classifier and the rule-based classifier
    rules_clf = RuleBasedTags(num_rules=6)
    praxi_clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args=vwargs, suffix=suffix, iterative=False,
                 use_temp_files=True)
    resfile = open(resfile_name, 'wb')
    results = []
    if(test_path==None): # cross val!
        all_wrong = []
        hyb_names = [f for f in listdir(train_path) if (isfile(join(train_path, f))and f[-3:]=='hyb')]

        #names_no_ext = [name[:-4] for name in tr_names]

        random.shuffle(hyb_names)
        #logging.info("Partitioning into %d folds", nfolds)
        folds = fold_partitioning(hyb_names, n=nfolds)
        #logging.info("Starting cross validation folds: ")

        for idx in range(len(folds)):
            # take current fold to be the "test", use the rest as training
            logging.info("Test fold is: %d", idx)
            #test_tagset_names = fold
            test_names = folds[idx]
            train_names_list = list(range(len(folds)))
            train_names_list.remove(idx)
            #logging.info("Training folds: %s", str(train_idx_list))
            train_names = []
            for i in train_names_list:
                train_names += folds[i]

            train_dics = parse_hyb(train_names, train_path)
            test_dics = parse_hyb(test_names, train_path)

            labels, preds, wrong_labs = get_scores(rules_clf, praxi_clf, train_dics, test_dics, outdir)
            all_wrong += wrong_labs
            results.append((labels, preds))

    else: # no folds/crossvalidation
        # get traintags, trainlabels, etc from ts_path, train_path
        ts_train_names = [f for f in listdir(train_path) if (isfile(join(train_path, f))and f[-3:]=='hyb')]
        ts_test_names = [f for f in listdir(ts_path) if (isfile(join(ts_path, f)) and f[-3:]=='hyb')]

        train_dics = parse_hyb(train_names, train_path)
        test_dics = parse_hyb(test_names, test_path)

        labels, preds, wrong_labs = get_scores(rules_clf, praxi_clf, train_dics, test_dics, outdir)
        results.append((labels, preds))

        #logging.info("Getting single label scores:")
        #labels, preds = get_scores(clf, train_tags, train_labels, test_tags, test_labels)

        #results.append((labels, preds))

    #all_wrong = list(dict.fromkeys(all_wrong))
    #print(all_wrong)
    pickle.dump(results, resfile)
    # results is a list of tuples!!
    resfile.close()
    logging.info("Printing results:")

    print_results(resfile_name, outdir, result_type)

def get_scores(rules_clf, praxi_clf, train_dics, test_dics, outdir):

    train_labels = []
    train_tags = []
    train_changes = []
    test_tags = []
    test_changes = []
    #test_labels = []

    # FIT PRAXI
    # separate into tags, labels lists
    for dic in train_dics:
        train_labels.append(dic['label'])
        train_tags.append(dic['tags'])
        train_changes.append(dic['changes'])
    for dic in test_dics:
        test_tags.append(dic['tags'])
        test_changes.append(dic['changes'])

    praxi_clf.fit(train_tags, train_labels)


    # FIT RULE-BASED
    # training set separated by CORRRECT label
    # DICTIONARY: key = app label, value = dic list
    rule_training_set = sep_dics(train_dics)

    rules_clf.fit_all(rule_training_set)

    print("Number of rules: ", rules_clf.total_rules)
    print("Number of apps: ", rules_clf.total_apps)
    print("Number of versions: ", rules_clf.total_versions)

    lab_preds = praxi_clf.predict(test_tags)

    wrong_app = 0
    for pred, lab in zip(lab_preds, train_labels):
        if pred != lab:
            wrong_app += 1
    print("Number of incorrect app names: ", wrong_app)

    # put predictions with test dictionaries
    for pred, dic in zip(lab_preds, test_dics):
        dic['label_prediction'] = pred

    # test set separated by label PREDICTION from Praxi
    separated_test_set = sep_dics(test_dics, guide='label_prediction')

    preds = rules_clf.predict_all(separated_test_set)

    # Now: evaluate performance!
    # turn into two lists of labels, preds
    full_labels = []
    full_preds = []
    apps = separated_test_set.keys()
    for app in apps:
        cur_dics = separated_test_set[app]
        cur_preds = preds[app]
        for cd, cp in zip (cur_dics, cur_preds):
            full_labels.append(cd['label'] + '.' + cd['version'])
            full_preds.append(app + '.' + cp)
    #print(full_preds)
    #input("Enter to continue...")

    wrong_labs = []
    for l, p in zip (full_labels, full_preds):
        if l != p:
            print(l,p)
            #just_lab = l.split('.')[0]
            #if just_lab not in wrong_labs:
            #    wrong_labs.append(just_lab)
    #print(wrong_labs)

    return full_labels, full_preds, wrong_labs

def print_results(resfile, outdir, result_type='summary', n_strats=1, args=None, iterative=False):
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
        if not iterative:
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
        else:
            file_header = (
                "ITERATIVE EXPERIMENTAL REPORT:\n" +
                time.strftime("Generated %c\n\n") +
                ('\nArgs: {}\n\n'.format(args) if args else '') +
                "LABEL COUNT : {}\n\n".format(lc) +
                "EXPERIMENT WITH {} TEST CHANGESETS\n".format(len(y_true)))
            fstub = 'iter_exp'

        os.makedirs(str(outdir), exist_ok=True) # makes directory if it doesn't exist
        if result_type == 'summary':
            fstub += '_summary'
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
            fname = get_free_filename(fstub, outdir, '.txt')
            f = open(fname, "w")
            f.write(file_header) # just need header b/c no confusion matrix
            f.close()
        else:
            # FULL RESULTS (original result format)
            file_header += (
                "F1 SCORE : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(f1w, f1i, f1a) +
                "PRECISION: {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(pw, pi, pa) +
                "RECALL   : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n#\n".format(rw, ri, ra))
            file_header += ("# {:-^55}\n#".format("CLASSIFICATION REPORT") + report.replace('\n', "\n#") +
                           " {:-^55}\n".format("CONFUSION MATRIX"))
            fname = get_free_filename(fstub, outdir, '.txt')
            savetxt("{}".format(fname),
                    confuse, fmt='%d', header=file_header, delimiter=',',comments='')

def sep_dics(dic_list, guide='label'):
    app_names = {}
    # get all names
    for dic in dic_list:
        # do stuff
        label = dic[guide]
        if label not in app_names:
            app_names[label] = []

    for dic in dic_list:
        label = dic[guide]
        app_names[label].append(dic)

    return app_names


if __name__ == '__main__':
    prog_start = time.time()

    parser = argparse.ArgumentParser(description='Arguments for Praxi software discovery algorithm.')
    parser.add_argument('-t','--traindir', help='Path to training tag/changeset directory.', default=None)
    parser.add_argument('-s', '--testdir', help='Path to testing train/tagset directory.', default=None)
    parser.add_argument('-o', '--outdir', help='Path to desired result directory', default='.')
    # run a single label experiment by default, if --multi flag is added, run a multilabel experiment!
    parser.add_argument('-w','--vwargs', dest='vw_args', default='-b 26 --learning_rate 1.5 --passes 10',
                        help="custom arguments for VW.")
    parser.add_argument('-n', '--nfolds', help='number of folds to use in cross validation', default=1) # make default 1?
    parser.add_argument('-f', '--fullres', help='generate full result file.', dest='result',
                        action='store_const', const='full', default='summary')
    parser.add_argument('-v', '--verbosity', dest='loglevel', action='store_const', const='DEBUG',
                        default='WARNING',help='specify level of detail for log file')
    # IMPLEMENT THIS!
    parser.add_argument('-l' '--labels', dest='print_labels', action='store_const', const=True, default=False,
                        help='Print missed labels')

    args = vars(parser.parse_args())

    outdir = os.path.abspath(args['outdir'])
    nfolds = int(args['nfolds'])

    train_path = args['traindir']
    test_path = args['testdir']

    #### SET UP LOGGING ####
    loglevel = args['loglevel']
    stub = 'praxi_exp'
    logfile_name = get_free_filename(stub, outdir, '.log')

    numeric_level = getattr(logging, loglevel, None)
    logging.basicConfig(filename=logfile_name,level=numeric_level)
    ########################

    # Log command line args
    result_type = args['result'] # full or summary
    logging.info("Result type: %s", result_type)

    print_misses = args['print_labels']

    vwargs = args['vw_args']
    logging.info("Arguments for Vowpal Wabbit: %s", vwargs)


    if(nfolds!= 1 and test_path!=None):
        # ERROR: SHOULDNT HAVE A TEST DIRECTORY IF CROSS VALIDATION IS OCCURRING
        logging.error("Too many input directories. If performing cross validation, expect just one.")
        raise ValueError("Too many input directories! Only need one for cross validation.")
    else:
        if(nfolds == 1): # no cv
            logging.info("Starting single label experiment")
            logging.info("Training directory: %s", ts_train_path)
            logging.info("Testing directory: %s", ts_test_path)
        else:
            # CROSS VALIDATION
            logging.info("Starting cross validation single label experiment with %s folds", str(nfolds))
            logging.info("Tagset directory: %s", train_path)
        resfile_name = get_free_filename('single_test', outdir, '.p') # add arg to set stub?
        single_label_experiment(nfolds, train_path, resfile_name, outdir, vwargs, result_type,
                                test_path=test_path, print_misses=print_misses) # no train directory

    logging.info("Program runtime: %s", str(time.time()-prog_start))
