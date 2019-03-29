# Imports
from collections import Counter
from hashlib import md5
from multiprocessing import Lock

import os
from os import listdir
from os.path import isfile, join

from pathlib import Path
import random
import tempfile
import time
import yaml
import pickle
import copy
import argparse

import envoy
from joblib import Memory
from sklearn.base import BaseEstimator
from tqdm import tqdm

import numpy as np
from numpy import savetxt
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from hybrid_tags import Hybrid

# Directory constants
PROJECT_ROOT = Path('~/praxi').expanduser() # leave as project root to access any necessary files
memory = Memory(cachedir='/home/ubuntu/caches/joblib-cache', verbose=0)
LABEL_DICT = Path('./pred_label_dict.pkl') # Do I need this?

LOCK = Lock()

#######################################
#   FUNCTIONS for accessing tagsets   #
#######################################
def parse_ts(tagset_names, ts_dir):
    # Arguments: - tagset_names: a list of names of tagsets
    #            - ts_dir: the directory in which they are located
    # Returns: - tags: list of lists-- tags for each tagset name
    #          - labels: application name corresponding to each tagset
    tags = []
    labels = []
    for ts_name in tqdm(tagset_names):
            ts_path = ts_dir + '/' + ts_name
            tagset = get_tagset(ts_path)
            if 'labels' in tagset:
                # Multilabel changeset
                labels.append(tagset['labels'])
            else:
                labels.append(tagset['label'])
            tags.append(tagset['tags'])
    return tags, labels

def get_tagset(ts_path): # combine with parse_ts
    # Argument: - complete path to a tagset .yaml file
    # Returns:  - tagset dictionary contained in file (tags, labels)
    with open(ts_path, 'r') as stream:
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


#################################
#### SINGLE LABEL EXPERIMENT ####
#################################

def single_label_experiment(test_tags, test_labels, train_tags, train_labels, resfile_name, outdir, vwargs):
    # instantiate hybrid object
    suffix = 'single'
    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args=vwargs, suffix=suffix, iterative=False,
                 use_temp_files=True)
    resfile = open(resfile_name, 'wb')
    results = []
    y_test, preds = get_scores(clf, train_tags, train_labels, test_tags, test_labels)
    # two lists
    results.append(y_test)
    results.append(preds)
    print(type(results))
    pickle.dump(results, resfile)
    resfile.close() # seems to work up to this point
    print_results(resfile_name, outdir, args=clf.get_args())

def get_scores(clf, X_train, y_train, X_test, y_test,
               binarize=False, human_check=False, store_true=False):
    """ Gets two lists of changeset ids, does training+testing """
    if binarize:
        binarizer = MultiLabelBinarizer()
        clf.fit(X_train, binarizer.fit_transform(y_train))
        preds = binarizer.inverse_transform(clf.predict(X_test))
    else:
        clf.fit(X_train, y_train) # train model
        preds = clf.predict(X_test) # predict labels for test set
        #if store_true:
        #    labels = clf.transform_labels(y_test)
        #    with open('/home/ubuntu/sets/true_labels.txt', 'w') as f:
        #        for label in labels:
        #            f.write(str(label) + '\n')
        #    logging.info("Wrote true labels to ~/sets/true_labels.txt")
    hits = misses = predictions = 0
    if LABEL_DICT.exists(): # only needed for human_check
        with LABEL_DICT.open('rb') as f:
            pred_label_dict = pickle.load(f)
    else:
        pred_label_dict = {}
    for pred, label in zip(preds, y_test):
        if human_check:
            while (pred, label) not in pred_label_dict:
                print("Does '{}' match the label '{}'? [Y/n]".format(pred, label))
                answer = input().lower()
                if answer == 'y':
                    pred_label_dict[(pred, label)] = True
                elif answer == 'n':
                    pred_label_dict[(pred, label)] = False
                else:
                    print("Please try again")
            with LABEL_DICT.open('wb') as f:
                pickle.dump(pred_label_dict, f)
            if pred_label_dict[(pred, label)]:
                hits += 1
            else:
                misses += 1
        else:
            if pred == label:
                hits += 1
            else:
                misses += 1
        predictions += 1
    # it looks like the hits, misses, and prediction vars are just for logging purposes... results are just "preds", and "y_test"
    #logging.info("Preds:" + str(predictions))
    #logging.info("Hits:" + str(hits))
    #logging.info("Misses:" + str(misses))
    return copy.deepcopy(y_test), preds

def print_results(resfile, outdir, n_strats=1, args=None, iterative=False):
    with open(resfile, 'rb') as f:
        results = pickle.load(f)
    y_true = results[0]
    y_pred = results[1]

    labels = sorted(set(y_true)) # gotta figure out this line...
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
    # y_true, y_pred used to be lists of lists
    #for x, y in zip(y_true, y_pred):
    x = y_true
    y = y_pred
    print("y_true length:", len(x))
    print("y_pred length:", len(y))
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

    for strat, report, f1w, f1i, f1a, pw, pi, pa, rw, ri, ra, confuse, lc in zip(
            range(n_strats), classifications, f1_weighted, f1_micro, f1_macro,
            p_weighted, p_micro, p_macro, r_weighted, r_micro, r_macro,
            confusions, label_counts):
        if not iterative:
            file_header = (
                "# SINGLE LABEL EXPERIMENTAL REPORT:\n" +
                time.strftime("# Generated %c\n#\n") +
                ('#\n# Args: {}\n#\n'.format(args) if args else '') +
                "NUMBER OF TESTING CHANGESETS: {}\n".format(len(y_pred)) )
            fname = get_free_filename('single_label_exp', outdir, '.txt')
        else:
            file_header = (
                "# ITERATIVE EXPERIMENTAL REPORT:\n" +
                time.strftime("# Generated %c\n#\n") +
                ('#\n# Args: {}\n#\n'.format(args) if args else '') +
                "# LABEL COUNT : {}\n".format(lc))
            fname = get_free_filename('iterative_exp', outdir, '.txt')
        # this part seemes fine
        file_header += (
            "# F1 SCORE : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(f1w, f1i, f1a) +
            "# PRECISION: {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(pw, pi, pa) +
            "# RECALL   : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n#\n".format(rw, ri, ra) +
            "# {:-^55}\n#".format("CLASSIFICATION REPORT") + report.replace('\n', "\n#") +
            " {:-^55}\n".format("CONFUSION MATRIX"))
        os.makedirs(str(outdir), exist_ok=True)
        # name the file something meaningful
        print(fname)
        savetxt("{}".format(fname),
                confuse, fmt='%d', header=file_header, delimiter=',',comments='')

# Make another print function that prints summary (F1 scores) and a side by side list of labels and predictions
def print_results_summary(resfile, outdir):
    print("Will make result summary file")

##############################################################################
###                MULTILABEL EXPERIMENTS                                   ##
##############################################################################

def multi_label_experiment(test_tags, test_labels, train_tags, train_labels, resfile_name, outdir, vwargs):
    suffix = 'multi'
    # VW ARGS SHOULD BE PASSED IN
    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=True,
                 vw_args=vwargs, suffix=suffix, use_temp_files=True)

    #logging.info("Prediction pickle is %s", resfile_name)
    resfile = open(resfile_name, 'wb')
    results = []
    y_test, preds = get_multilabel_scores(clf, train_tags, train_labels, test_tags, test_labels)
    results.append(y_test)
    results.append(preds)
    pickle.dump(results, resfile)
    resfile.close()
    print_multilabel_results(resfile_name, outdir, args=clf.get_args(), n_strats=1)

def get_multilabel_scores(clf, X_train, y_train, X_test, y_test):
    """Gets scores while providing the ntags to clf"""
    clf.fit(X_train, y_train)
    # rulefile = get_free_filename('rules', '.', suffix='.yml')
    # logging.info("Dumping rules to %s", rulefile)
    # with open(rulefile, 'w') as f:
    #     yaml.dump(clf.rules, f)
    ntags = [len(y) if isinstance(y, list) else 1 for y in y_test]
    preds = clf.top_k_tags(X_test, ntags)
    #hits = misses = predictions = 0
    #for pred, label in zip(preds, y_test):
    #    if pred == label:
    #        hits += 1
    #    else:
    #        misses += 1
    #    predictions += 1
    #logging.info("Preds:" + str(predictions)) # Use for result summary file???
    #logging.info("Hits:" + str(hits))
    #logging.info("Misses:" + str(misses))
    return y_test, preds

def print_multilabel_results(resfile, outdir, args=None, n_strats=1):
    #logging.info('Writing scores to %s', str(outdir))
    with open(resfile, 'rb') as f:
        results = pickle.load(f)
    y_true = results[0]
    y_pred = results[1]

    bnz = MultiLabelBinarizer()
    bnz.fit(y_true)
    all_tags = copy.deepcopy(y_true)
    for preds in y_pred:
        for label in preds:
            if label not in bnz.classes_:
                all_tags.append([label])
                bnz.fit(all_tags)
    y_true = bnz.transform(y_true)
    y_pred = bnz.transform(y_pred)

    labels = bnz.classes_
    report = metrics.classification_report(y_true, y_pred, target_names=labels)
    f1w = metrics.f1_score(y_true, y_pred, average='weighted')
    f1i = metrics.f1_score(y_true, y_pred, average='micro')
    f1a = metrics.f1_score(y_true, y_pred, average='macro')
    pw = metrics.precision_score(y_true, y_pred, average='weighted')
    pi = metrics.precision_score(y_true, y_pred, average='micro')
    pa = metrics.precision_score(y_true, y_pred, average='macro')
    rw = metrics.recall_score(y_true, y_pred, average='weighted')
    ri = metrics.recall_score(y_true, y_pred, average='micro')
    ra = metrics.recall_score(y_true, y_pred, average='macro')

    file_header = (
        "# MULTILABEL EXPERIMENT REPORT\n" +
        time.strftime("# Generated %c\n#\n") +
        ('#\n# Args: {}\n#\n'.format(args) if args else '') +
        "# 3 FOLD CROSS VALIDATION WITH {} CHANGESETS\n".format(len(y_true)) +
        "# F1 SCORE : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(f1w, f1i, f1a) +
        "# PRECISION: {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(pw, pi, pa) +
        "# RECALL   : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n#\n".format(rw, ri, ra) +
        "# {:-^55}\n#".format("CLASSIFICATION REPORT") + report.replace('\n', "\n#"))

    # create file name
    fname = get_free_filename('multi_exp', outdir, '.txt')

    os.makedirs(str(outdir), exist_ok=True)
    savetxt("{}".format(fname),
            np.array([]), fmt='%d', header=file_header, delimiter=',',
            comments='')

    # Write in a file! ^^

def print_multi_results_summary(resfile, outdir):
    print("Will make result summary file")

if __name__ == '__main__':

    prog_start = time.time()

    #ts_train_path = '/home/ubuntu/praxi/results/week7/tagsets_train'
    #ts_test_path = '/home/ubuntu/praxi/results/week7/tagsets_test'

    ##############################################################################
    ### GET TEST AND TRAIN TAGSETS FROM DIRECTORIES INPUT AS COMMAND LINE FLAGS ##
    ##############################################################################

    parser = argparse.ArgumentParser(description='Arguments for Praxi software discovery algorithm.')
    parser.add_argument('-tr','--traindir', help='Path to training tagset directory.', required=True)
    parser.add_argument('-ts', '--testdir', help='Path to testing tagset directoy.', required=True)
    parser.add_argument('-od', '--outdir', help='Path to desired result directory', default='.')
    # run a single label experiment by default, if --multi flag is added, run a multilabel experiment!
    parser.add_argument('--multi', dest='experiment', action='store_const', const='multi', default='single', help="Type of experiment to run (single-label default).")
    parser.add_argument('-vw','--vwargs', dest='vw_args', default='-b 26 --learning_rate 1.5 --passes 10', help="custom arguments for VW.")

    #'-b 26 --learning_rate 1.5 --passes 10'
    # default single, arg to do multi instead

    #parser.add_argument('--traindir', dest='accumulate', action='store_const',
    #               const=sum, default=max,
    #               help='sum the integers (default: find the max)')

    args = vars(parser.parse_args())
    print(args)

    exp_type = args['experiment'] # 'single' or 'multi'
    vwargs = args['vw_args']
    outdir = os.path.abspath(args['outdir'])
    # also put pickle file in outdir

    ts_train_path = args['traindir']
    print(ts_train_path)
    ts_train_names = [f for f in listdir(ts_train_path) if (isfile(join(ts_train_path, f))and f[-3:]=='tag')]

    ts_test_path = args['testdir']
    print(ts_test_path)
    ts_test_names = [f for f in listdir(ts_test_path) if (isfile(join(ts_test_path, f)) and f[-3:]=='tag')]
    # ^ Make sure to filter specifically for tagsets!! (Should end with .tag extension)

    train_tags, train_labels = parse_ts(ts_train_names, ts_train_path)
    test_tags, test_labels = parse_ts(ts_test_names, ts_test_path)
    # ^^ This is what I will pass into the clf.fit function


    ##############################################################################
    ### RUN EXPERIMENT ###########################################################
    ##############################################################################
    if exp_type == 'single':
        print('single')
        resfile_name = get_free_filename('single_test', outdir, '.p')
        # Run single-label experiment
        # 1) train model
        # 2) predict labels
        # 3) write results to a file
        single_label_experiment(test_tags, test_labels, train_tags, train_labels, resfile_name, outdir, vwargs)
    else:
        print('multi')
        resfile_name = get_free_filename('multi_test', outdir, '.p')
        multi_label_experiment(test_tags, test_labels, train_tags, train_labels, resfile_name, outdir, vwargs)

        # Run multi-label experiment

    print("Program runtime:", (time.time() - prog_start))


    # functions in hybrid_tags.py to test:
    # - score: should work if predict works
