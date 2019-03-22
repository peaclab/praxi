#!/usr/bin/env python3

# Imports
from collections import Counter
from hashlib import md5
from multiprocessing import Lock
import os
from pathlib import Path
import random
import tempfile
import time
import yaml
import pickle
import copy

import envoy
from joblib import Memory
from sklearn.base import BaseEstimator
from tqdm import tqdm

import numpy as np
from numpy import savetxt
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from hybrid import Hybrid
from hybrid import Columbus
#from rule_based import RuleBased

from columbus.columbus import columbus
from columbus.columbus import refresh_columbus

# Directory constants
PROJECT_ROOT = Path('~/praxi').expanduser() # leave as project root to access any necessary files
CHANGESET_ROOT = Path('~/caches/changesets/').expanduser()
COLUMBUS_CACHE = Path('~/caches/columbus-cache-2').expanduser()
memory = Memory(cachedir='/home/ubuntu/caches/joblib-cache', verbose=0)
LABEL_DICT = Path('./pred_label_dict.pkl')
#RESULT_DIR = Path('~/praxi/week4/results').expanduser()

LOCK = Lock()

def get_free_filename(stub, directory, suffix=''):
    counter = 0
    while True:
        file_candidate = '{}/{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            counter += 1
        else:  # No match found
            if suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate

@memory.cache
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
        #logging.error("Malformed changeset, id: %d, changeset: %s",
        #              csid, csfile)
        raise IOError("Couldn't read changeset")
    return changeset


def multiapp_trainw_dirty():
    resfile_name = get_free_filename('multilabel_rerun', '.', suffix='.pkl')
    outdir = get_free_filename('multiapp_newparams', '/home/ubuntu/praxi/week7/multi_res')
    suffix = 'timing' # what is this for?
    # clf = RuleBased(filter_method='take_max', num_rules=6)
    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=True,
                 vw_args='-b 26 --passes 50 --bfgs --learning_rate 1.25 --decay_learning_rate 0.9',
                 suffix=suffix, use_temp_files=True)

    #print(clf.get_args())
    with (PROJECT_ROOT / 'changeset_sets' /
          'multilabel_chunks.p').open('rb') as f:
        multilabel_chunks = pickle.load(f) # three groups of 1000

    # Get single app dirty changesets
    with (PROJECT_ROOT / 'changeset_sets' /
          'threek_dirty_chunks.p').open('rb') as f:
        threeks = pickle.load(f) # three groups of 1000

    #logging.info("Prediction pickle is %s", resfile_name)
    # three-fold cross validation
    resfile = open(resfile_name, 'wb')
    results = []
    for ml_idx, ml_chunk in enumerate(multilabel_chunks):
        #logging.info('Test set is %d', ml_idx)
        ml_train_idx = [0, 1, 2]
        ml_train_idx.remove(ml_idx)
        X_train, y_train = parse_csids(multilabel_chunks[ml_train_idx[0]],
                                       multilabel=True)
        features, labels = parse_csids(multilabel_chunks[ml_train_idx[1]],
                                       multilabel=True)
        X_train += features
        y_train += labels
        X_test, y_test = parse_csids(ml_chunk, multilabel=True) # test is the one that wasn't removed
        results.append(get_multilabel_scores(clf, X_train, y_train, X_test, y_test))
        print("iteration complete")
        pickle.dump(results, resfile)
        resfile.seek(0)

        # adding each extra label
        for idx, chunk in tqdm(enumerate(threeks)):
            print('Extra training set is ', idx)
            features, labels = parse_csids(chunk, multilabel=True)
            X_train += features
            y_train += labels
            results.append(get_multilabel_scores(clf, X_train, y_train, X_test, y_test))
            pickle.dump(results, resfile)
            resfile.seek(0)
    resfile.close()
    print_multilabel_results(resfile_name, outdir, args=clf.get_args(), n_strats=4)


def print_multilabel_results(resfile, outdir, args=None, n_strats=1):
    #logging.info('Writing scores to %s', str(outdir))
    with open(resfile, 'rb') as f:
        results = pickle.load(f)
    # # Now do the evaluation!
    # #results = [
    # #    0 => ([x, y, z], <-- true
    # #          [x, y, k]) <-- pred
    # #]
    y_trues = [[] for _ in range(n_strats)]
    y_preds = [[] for _ in range(n_strats)]
    for idx, result in enumerate(results):
        y_trues[idx % n_strats] += result[0]
        y_preds[idx % n_strats] += result[1]

    for strat, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
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
            "# {:-^55}\n#".format("CLASSIFICATION REPORT") + report.replace('\n', "\n#")
        )
        os.makedirs(str(outdir), exist_ok=True) # create a directory
        savetxt("{}/{}.txt".format(outdir, strat),
                np.array([]), fmt='%d', header=file_header, delimiter=',',
                comments='')

def get_multilabel_scores(clf, X_train, y_train, X_test, y_test):
    """Gets scores while providing the ntags to clf"""
    print("Training model")
    clf.fit(X_train, y_train)
    # rulefile = get_free_filename('rules', '.', suffix='.yml')
    # logging.info("Dumping rules to %s", rulefile)
    # with open(rulefile, 'w') as f:
    #     yaml.dump(clf.rules, f)
    ntags = [len(y) if isinstance(y, list) else 1 for y in y_test]
    preds = clf.top_k_tags(X_test, ntags)
    hits = misses = predictions = 0
    for pred, label in zip(preds, y_test):
        if pred == label:
            hits += 1
        else:
            misses += 1
        predictions += 1
    #logging.info("Preds:" + str(predictions))
    #logging.info("Hits:" + str(hits))
    #logging.info("Misses:" + str(misses))
    return y_test, preds

if __name__ == '__main__':
    #setup_logging()
    # resfile_name = './results-multiapp-hybrid-1.pkl'
    # outdir = get_free_filename('hybrid-results-multiapp', '/home/centos/results')
    # print_multilabel_results('./results-rule-0.pkl', '/home/centos/results/rule0',
    #                          n_strats=4)
    # print_multilabel_results('./results-rule-1.pkl', '/home/centos/results/rule1',
    #                          n_strats=4)
    multiapp_trainw_dirty()
    print("FINISHED!!!!!!")
    #resfile_name = "multilabel_rerun-0.pkl"
    #outdir = "/home/ubuntu/praxi/week6/multi_res"
    #print_multilabel_results(resfile_name, outdir, args=None, n_strats=4) # got rid of 'args', but just used in header so fine
    # iterative_tests()
    # onekdirty()

# RUN
