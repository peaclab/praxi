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

#import envoy
#from joblib import Memory
from sklearn.base import BaseEstimator
from tqdm import tqdm

import numpy as np
from numpy import savetxt
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from hybrid import Hybrid

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

def parse_csids(csids, multilabel=False, iterative=False):
    """ Returns labels and features from csids, features are file sets
    file sets: list of string of format '644 /usr/.../file' """
    features = []
    labels = []
    for csid in tqdm(csids):
        changeset = get_changeset(csid, iterative=iterative)
        if multilabel:
            if 'labels' in changeset:
                labels.append(changeset['labels'])
            else:
                labels.append([changeset['label']])
        else:
            labels.append(changeset['label'])
        features.append(changeset['changes'])
    return features, labels

def get_changeset(csid, iterative=False):
    changeset = None
    if str(csid) in {'5', '6', '7'}:
        # Dirty fix for finger, autotrace
        globstr = '*[!16].5'
    else:
        globstr = '*.{}'.format(csid)
    if iterative:
        globstr += '.yaml'
    else:
        globstr += '.[!y]*'
    for csfile in CHANGESET_ROOT.glob(globstr):
        if changeset is not None:
            raise IOError(
                "Too many changesets match the csid {}, globstr {}".format(
                    csid, globstr))
        with csfile.open('r') as f:
            changeset = yaml.load(f)
    if changeset is None:
        raise IOError("No changesets match the csid {}".format(csid))
    if 'changes' not in changeset or (
            'label' not in changeset and 'labels' not in changeset):
        logging.error("Malformed changeset, id: %d, changeset: %s",
                      csid, csfile)
        raise IOError("Couldn't read changeset")
    return changeset


def iterative_tests():
    # get result file name
    outdir = '/home/ubuntu/praxi/results/iterative'
    vwargs = '-b 26 --learning_rate 1.5 --passes 10'
    resfile_name = get_free_filename('iterative-hybrid', outdir, suffix='.pkl')
    suffix = 'hybrid'
    iterative = True
    # clf = RuleBased(filter_method='take_max', num_rules=6)
    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args= vwargs, suffix=suffix, iterative=iterative,
                 use_temp_files=True)
    # clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True,
    #              suffix=suffix,
    #              probability=True, tqdm=True)
    # Get single app dirty changesets

    with (Path('/home/ubuntu/praxi/changeset_sets/').expanduser() / 'iterative_chunks.p').open('rb') as f:
        it_chunks = pickle.load(f)

    #print("Prediction pickle is %s", resfile_name)
    resfile = open(resfile_name, 'wb')
    results = []
    for i in range(3):
        i1 = i % 3
        i2 = (i + 1) % 3
        i3 = (i + 2) % 3
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        clf.refresh()
        for idx, inner_chunks in enumerate(it_chunks):
            print('In iteration %d', idx)
            features, labels = parse_csids(inner_chunks[i1], iterative=True)
            if iterative:
                X_train = features
                y_train = labels
            else:
                X_train += features
                y_train += labels
            features, labels = parse_csids(inner_chunks[i2], iterative=True)
            X_train += features
            y_train += labels
            features, labels = parse_csids(inner_chunks[i3], iterative=True)
            X_test += features
            y_test += labels
            results.append(get_scores(clf, X_train, y_train, X_test, y_test))
            pickle.dump(results, resfile)
            resfile.seek(0)
    resfile.close()
    print_results(resfile_name, outdir, args=clf.get_args(),
                  n_strats=len(it_chunks), iterative=True)


def get_scores(clf, X_train, y_train, X_test, y_test,
               binarize=False):
    """ Gets two lists of changeset ids, does training+testing """
    if binarize:
        binarizer = MultiLabelBinarizer()
        clf.fit(X_train, binarizer.fit_transform(y_train))
        preds = binarizer.inverse_transform(clf.predict(X_test))
    else:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
    #if LABEL_DICT.exists():
    #    with LABEL_DICT.open('rb') as f:
    #        pred_label_dict = pickle.load(f)
    #else:
    #    pred_label_dict = {}
    #for pred, label in zip(preds, y_test):
    return copy.deepcopy(y_test), preds

def print_results(resfile, outdir, n_strats=5, args=None, iterative=False):
    print('Writing scores to %s', str(outdir))
    with open(resfile, 'rb') as f:
        results = pickle.load(f)
    # # Now do the evaluation!
    # #results = [
    # #    0 => ([x, y, z], <-- true
    # #          [x, y, k]) <-- pred
    # #]
    y_true = [[] for _ in range(n_strats)]
    y_pred = [[] for _ in range(n_strats)]
    for idx, result in enumerate(results):
        y_true[idx % n_strats] += result[0]
        y_pred[idx % n_strats] += result[1]

    labels = sorted(set(j for i in range(n_strats) for j in y_true[i]))
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
    for x, y in zip(y_true, y_pred):
        classifications.append(metrics.classification_report(x, y, labels))
        f1_weighted.append(metrics.f1_score(x, y, labels, average='weighted'))
        f1_micro.append(metrics.f1_score(x, y, labels, average='micro'))
        f1_macro.append(metrics.f1_score(x, y, labels, average='macro'))
        p_weighted.append(
            metrics.precision_score(x, y, labels, average='weighted'))
        p_micro.append(metrics.precision_score(x, y, labels, average='micro'))
        p_macro.append(metrics.precision_score(x, y, labels, average='macro'))
        r_weighted.append(
            metrics.recall_score(x, y, labels, average='weighted'))
        r_micro.append(metrics.recall_score(x, y, labels, average='micro'))
        r_macro.append(metrics.recall_score(x, y, labels, average='macro'))
        confusions.append(metrics.confusion_matrix(x, y, labels))
        label_counts.append(len(set(x)))

    for strat, report, f1w, f1i, f1a, pw, pi, pa, rw, ri, ra, confuse, lc in zip(
            range(n_strats), classifications, f1_weighted, f1_micro, f1_macro,
            p_weighted, p_micro, p_macro, r_weighted, r_micro, r_macro,
            confusions, label_counts):
        if not iterative:
            clean_tr_str = (
                "nil = 1000 total" if strat == 0 else
                ','.join(str(x) for x in range(strat)) + " ({}) = {} total".format(
                    strat * 2500, strat * 2500 + 1000))
            file_header = (
                "# 1K DIRTY EXPERIMENTAL REPORT: STRATUM {}\n".format(strat) +
                time.strftime("# Generated %c\n#\n") +
                ('#\n# Args: {}\n#\n'.format(args) if args else '') +
                "# TRAIN: Dirty (1000, avg'ed over #0,1,2) + Clean #{}\n".format(clean_tr_str) +
                "# TEST : Dirty (2000, avg'ed over #0,1,2) + Clean #nil = 2000 total\n")
        else:
            file_header = (
                "# ITERATIVE EXPERIMENTAL REPORT: STRATUM {}\n".format(strat) +
                time.strftime("# Generated %c\n#\n") +
                ('#\n# Args: {}\n#\n'.format(args) if args else '') +
                "# LABEL COUNT : {}\n".format(lc))
        file_header += (
            "# F1 SCORE : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(f1w, f1i, f1a) +
            "# PRECISION: {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(pw, pi, pa) +
            "# RECALL   : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n#\n".format(rw, ri, ra) +
            "# {:-^55}\n#".format("CLASSIFICATION REPORT") + report.replace('\n', "\n#") +
            " {:-^55}\n".format("CONFUSION MATRIX")
        )
        os.makedirs(str(outdir), exist_ok=True)
        savetxt("{}/{}.txt".format(outdir, strat),
                confuse, fmt='%d', header=file_header, delimiter=',',
                comments='')


if __name__ == '__main__':
    iterative_tests()
