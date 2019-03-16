#!/usr/bin/env python3

import copy
import logging
import logging.config
import itertools
import os
import pickle
from pathlib import Path
import random
import time
import yaml

from tqdm import tqdm #makes loop show a progress meter??
import numpy as np
from numpy import savetxt
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from joblib import Memory

from hybrid import Hybrid
from hybrid import Columbus
#from rule_based import RuleBased

PROJECT_ROOT = Path('~/praxi').expanduser()
CHANGESET_ROOT = Path('~/caches/changesets/').expanduser()
memory = Memory(cachedir='/home/ubuntu/caches/joblib-cache', verbose=0)
LABEL_DICT = Path('./pred_label_dict.pkl')


"""
def iterative_tests():
    resfile_name = get_free_filename('iterative-hybrid', '.', suffix='.pkl')
    outdir = get_free_filename('iterative-hybrid', '/home/centos/results')
    suffix = 'hybrid'
    iterative = False
    # clf = RuleBased(filter_method='take_max', num_rules=6)
    # Declare praxi object
    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args='-b 26 --learning_rate 1.5',
                 suffix=suffix, iterative=iterative,
                 use_temp_files=True
                 )
    # clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True,
    #              suffix=suffix,
    #              probability=True, tqdm=True)
    # Get single app dirty changesets
    with (PROJECT_ROOT / 'iterative_chunks.p').open('rb') as f:
        it_chunks = pickle.load(f)

    logging.info("Prediction pickle is %s", resfile_name)
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
            logging.info('In iteration %d', idx)
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
"""


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


def multiapp_trainw_dirty():
    resfile_name = get_free_filename('multilabel_rerun', '.', suffix='.pkl')
    outdir = get_free_filename('multiapp_2', '/home/ubuntu/praxi/week6/multi_res')
    suffix = 'timing' # what is this for?
    # clf = RuleBased(filter_method='take_max', num_rules=6)
    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=True,
                 vw_args='-b 26 --learning_rate 1.5 --passes 10',
                 suffix=suffix, use_temp_files=True)
    # Get multiapp changesets
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
        results.append(get_multilabel_scores(
            clf, X_train, y_train, X_test, y_test))
        pickle.dump(results, resfile)
        resfile.seek(0)

        # adding each extra label
        for idx, chunk in tqdm(enumerate(threeks)):
            print('Extra training set is ', idx)
            features, labels = parse_csids(chunk, multilabel=True)
            X_train += features
            y_train += labels
            results.append(get_multilabel_scores(
                clf, X_train, y_train, X_test, y_test))
            pickle.dump(results, resfile)
            resfile.seek(0)
    resfile.close()
    #return resfile_name, outdir
    print_multilabel_results(resfile_name, outdir, args=clf.get_args(), n_strats=4)


"""def onekdirty():
    resfile_name = './results-timing-rule.pkl'
    outdir = 'rule-timing-results'
    suffix = 'rule-timing'
    clf = RuleBased(filter_method='take_max', num_rules=6)
    # clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
    #              vw_args='-b 26 --learning_rate 1.5 --passes 10',
    #              suffix=suffix, iterative=False,
    #              use_temp_files=True
    #              )
    with (PROJECT_ROOT / 'changeset_sets' /
          'threek_dirty_chunks.p').open('rb') as f:
        threeks = pickle.load(f)
    with (PROJECT_ROOT / 'changeset_sets' /
          'tenk_clean_chunks.p').open('rb') as f:
        tenks = pickle.load(f)
    resfile = open(resfile_name, 'wb')
    results = []
    for idx, train_csids in tqdm(enumerate(threeks)):
        #logging.info('Train set is %d', idx)
        test_idx = [0, 1, 2]
        test_idx.remove(idx)
        # Split calls to parse_csids for more efficient memoization
        X_test, y_test = parse_csids(threeks[test_idx[0]])
        features, labels = parse_csids(threeks[test_idx[1]])
        X_test += features
        y_test += labels
        X_train, y_train = parse_csids(train_csids)
        results.append(get_scores(clf, X_train, y_train,
                                  X_test, y_test))
        pickle.dump(results, resfile)
        resfile.seek(0)
        for inner_idx, extra_cleans in tqdm(enumerate(tenks)):
            #logging.info('Extra clean count: %d', inner_idx + 1)
            features, labels = parse_csids(extra_cleans)
            X_train += features
            y_train += labels
            results.append(get_scores(clf, X_train, y_train,
                                      X_test, y_test))
            pickle.dump(results, resfile)
            resfile.seek(0)
    resfile.close()
    print_results(resfile_name, outdir)
"""

def clean_test():
    outdir = 'result-rule-clean'
    with (PROJECT_ROOT / 'changeset_sets' /
          'tenk_clean_chunks.p').open('rb') as f:
        tenks = pickle.load(f)
    resfile = open('./results-rule-clean.pkl', 'wb')
    results = []
    for idx, test_csids in tqdm(enumerate(tenks)):
        #logging.info('Test set is %d', idx)
        train_idx = list(range(len(tenks)))
        train_idx.remove(idx)
        # Split calls to parse_csids for more efficient memoization
        # Splitting changesets into train/test groups
        X_train = []
        y_train = []
        train_csids = []
        for train_i in train_idx:
            features, labels = parse_csids(tenks[train_i])
            X_train += features
            y_train += labels
            train_csids += tenks[train_i]
        X_test, y_test = parse_csids(tenks[idx])
        results.append(get_scores(X_train, y_train,
                                  X_test, y_test)) # does this even work? no training object
        pickle.dump(results, resfile)
        resfile.seek(0)
    resfile.close()
    print_results('./results-rule-clean.pkl', outdir)


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


def print_results(resfile, outdir, n_strats=5, args=None, iterative=False):
    #logging.info('Writing scores to %s', str(outdir))
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


def get_multilabel_scores(clf, X_train, y_train, X_test, y_test):
    """Gets scores while providing the ntags to clf"""
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


"""
def get_scores(clf, X_train, y_train, X_test, y_test,
               binarize=False, human_check=False, store_true=False):
    "" Gets two lists of changeset ids, does training+testing "
    if binarize:
        binarizer = MultiLabelBinarizer()
        clf.fit(X_train, binarizer.fit_transform(y_train))
        preds = binarizer.inverse_transform(clf.predict(X_test))
    else:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        if store_true:
            labels = clf.transform_labels(y_test)
            with open('/home/ubuntu/sets/true_labels.txt', 'w') as f:
                for label in labels:
                    f.write(str(label) + '\n')
            #logging.info("Wrote true labels to ~/sets/true_labels.txt")
    hits = misses = predictions = 0
    if LABEL_DICT.exists():
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
    #logging.info("Preds:" + str(predictions))
    #logging.info("Hits:" + str(hits))
    #logging.info("Misses:" + str(misses))
    return copy.deepcopy(y_test), preds
"""

# sets up logging for execution
# GET EXPLANATION
"""
def setup_logging():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'standard': {
                'format': '%(asctime)s %(levelname)-7s %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    })
"""

if __name__ == '__main__':
    #setup_logging()
    # resfile_name = './results-multiapp-hybrid-1.pkl'
    # outdir = get_free_filename('hybrid-results-multiapp', '/home/centos/results')
    # print_multilabel_results('./results-rule-0.pkl', '/home/centos/results/rule0',
    #                          n_strats=4)
    # print_multilabel_results('./results-rule-1.pkl', '/home/centos/results/rule1',
    #                          n_strats=4)
    #multiapp_trainw_dirty()
    resfile_name = "multilabel_rerun-0.pkl"
    outdir = "/home/ubuntu/praxi/week6/multi_res"
    print_multilabel_results(resfile_name, outdir, args=None, n_strats=4) # got rid of 'args', but just used in header so fine
    # iterative_tests()
    # onekdirty()

# RUN 
