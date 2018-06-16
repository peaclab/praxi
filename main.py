#!/usr/bin/env python3

import copy
import logging
import logging.config
import pickle
from pathlib import Path
import random
import time
import yaml

from tqdm import tqdm
from numpy import savetxt
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from joblib import Memory

from hybrid import Hybrid
from rule_based import RuleBased

PROJECT_ROOT = Path('~/hybrid-method').expanduser()
CHANGESET_ROOT = Path('~/caches/multiapp/').expanduser()
memory = Memory(cachedir='/home/ubuntu/caches/joblib-cache', verbose=0)


def multiapp():
    resfile_name = './results-multiapp-hybrid.pkl'
    outdir = 'hybrid-results-multiapp'
    clf = OneVsRestClassifier(Hybrid())
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
    csids = []
    with open('/home/ubuntu/multi_app/changesets.txt', 'r') as f:
        for line in f:
            csids.append(int(line.strip()))
    random.seed(51)
    random.shuffle(csids)
    nfolds = 3
    fold_size = len(csids) // nfolds
    chunks = []
    for i in range(nfolds):
        chunks.append(csids[fold_size * i:fold_size * (i+1)])

    resfile = open(resfile_name, 'wb')
    results = []
    for idx, chunk in tqdm(enumerate(chunks)):
        test_csids = copy.deepcopy(chunk)
        logging.info('Test set is %d', idx)
        train_idx = list(range(nfolds))
        train_idx.remove(idx)
        # Split calls to parse_csids for more efficient memoization
        X_train, y_train = parse_csids(chunks[train_idx[0]], multilabel=True)
        features, labels = parse_csids(chunks[train_idx[1]], multilabel=True)
        X_train += features
        y_train += labels
        X_test, y_test = parse_csids(test_csids, multilabel=True)
        train_csids = chunks[train_idx[0]] + chunks[train_idx[1]]
        results.append(get_scores(clf, X_train, y_train, train_csids,
                                  X_test, y_test, test_csids))
        pickle.dump(results, resfile)
        resfile.seek(0)
    resfile.close()
    print_results(resfile_name, outdir)


def onekdirty():
    resfile_name = './results-hybrid.pkl'
    outdir = 'hybrid-results'
    clf = Hybrid()
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
    with (PROJECT_ROOT / 'changeset_sets' /
          'threek_dirty_chunks.p').open('rb') as f:
        threeks = pickle.load(f)
    with (PROJECT_ROOT / 'changeset_sets' /
          'tenk_clean_chunks.p').open('rb') as f:
        tenks = pickle.load(f)
    resfile = open(resfile_name, 'wb')
    results = []
    for idx, chunk in tqdm(enumerate(threeks)):
        train_csids = copy.deepcopy(chunk)
        logging.info('Train set is %d', idx)
        test_idx = [0, 1, 2]
        test_idx.remove(idx)
        # Split calls to parse_csids for more efficient memoization
        X_test, y_test = parse_csids(threeks[test_idx[0]])
        features, labels = parse_csids(threeks[test_idx[1]])
        X_test += features
        y_test += labels
        X_train, y_train = parse_csids(train_csids)
        test_csids = threeks[test_idx[0]] + threeks[test_idx[1]]
        results.append(get_scores(clf, X_train, y_train, train_csids,
                                  X_test, y_test, test_csids))
        pickle.dump(results, resfile)
        resfile.seek(0)
        for inner_idx, extra_cleans in tqdm(enumerate(tenks)):
            logging.info('Extra clean count: %d', inner_idx + 1)
            features, labels = parse_csids(extra_cleans)
            X_train += features
            y_train += labels
            train_csids += extra_cleans
            results.append(get_scores(clf, X_train, y_train, train_csids,
                                      X_test, y_test, test_csids))
            pickle.dump(results, resfile)
            resfile.seek(0)
    resfile.close()
    print_results(resfile_name, outdir)


def clean_test():
    outdir = 'result-rule-clean'
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
    with (PROJECT_ROOT / 'changeset_sets' /
          'tenk_clean_chunks.p').open('rb') as f:
        tenks = pickle.load(f)
    resfile = open('./results-rule-clean.pkl', 'wb')
    results = []
    for idx, test_csids in tqdm(enumerate(tenks)):
        logging.info('Test set is %d', idx)
        train_idx = list(range(len(tenks)))
        train_idx.remove(idx)
        # Split calls to parse_csids for more efficient memoization
        X_train = []
        y_train = []
        train_csids = []
        for train_i in train_idx:
            features, labels = parse_csids(tenks[train_i])
            X_train += features
            y_train += labels
            train_csids += tenks[train_i]
        X_test, y_test = parse_csids(tenks[idx])
        results.append(get_scores(X_train, y_train, train_csids,
                                  X_test, y_test, test_csids))
        pickle.dump(results, resfile)
        resfile.seek(0)
    resfile.close()
    print_results('./results-rule-clean.pkl', outdir)


def print_results(resfile, outdir, n_strats=5):
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

    for strat, report, f1w, f1i, f1a, pw, pi, pa, rw, ri, ra, confuse in zip(
            range(n_strats), classifications, f1_weighted, f1_micro, f1_macro,
            p_weighted, p_micro, p_macro, r_weighted, r_micro, r_macro,
            confusions):
        clean_tr_str = (
            "nil = 1000 total" if strat == 0 else
            ','.join(str(x) for x in range(strat)) + " ({}) = {} total".format(
                strat * 2500, strat * 2500 + 1000))
        file_header = (
            "# 1K DIRTY EXPERIMENTAL REPORT: STRATUM {}\n".format(strat) +
            time.strftime("# Generated %c\n#\n") +
            "# TRAIN: Dirty (1000, avg'ed over #0,1,2) + Clean #{}\n".format(clean_tr_str) +
            "# TEST : Dirty (2000, avg'ed over #0,1,2) + Clean #nil = 2000 total\n" +
            "# F1 SCORE : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(f1w, f1i, f1a) +
            "# PRECISION: {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(pw, pi, pa) +
            "# RECALL   : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n#\n".format(rw, ri, ra) +
            "# {:-^55}\n#".format("CLASSIFICATION REPORT") + report.replace('\n', "\n#") +
            " {:-^55}\n".format("CONFUSION MATRIX")
        )
        savetxt("/home/ubuntu/{}/{}.txt".format(outdir, strat),
                confuse, fmt='%d', header=file_header, delimiter=',',
                comments='')


def get_changeset(csid):
    changeset = None
    if str(csid) in {'5', '6', '7'}:
        # Dirty fix for finger, autotrace
        globstr = '*[!16].5.*'
    else:
        globstr = '*.{}.*'.format(csid)
    for csfile in CHANGESET_ROOT.glob(globstr):
        if changeset is not None:
            raise IOError(
                "Too many changesets match the csid {}, globstr {}".format(
                    csid, globstr))
        with csfile.open('r') as f:
            changeset = yaml.load(f)
    if changeset is None:
        raise IOError("No changesets match the csid {}".format(csid))
    if 'label' not in changeset or 'changes' not in changeset:
        logging.error("Malformed changeset, id: %d, changeset: %s",
                      csid, csfile)
        raise IOError("Couldn't read changeset")
    return changeset


@memory.cache
def parse_csids(csids, multilabel=False):
    """ Returns labels and features from csids, features are file sets
    file sets: list of string of format '644 /usr/.../file' """
    features = []
    labels = []
    for csid in tqdm(csids):
        changeset = get_changeset(csid)
        if multilabel:
            labels.append(changeset['labels'])
        else:
            labels.append(changeset['label'])
        features.append(changeset['changes'])
    return features, labels


def get_scores(clf, X_train, y_train, csids_train, X_test, y_test, csids_test):
    """ Gets two lists of changeset ids, does training+testing """
    clf.fit(X_train, y_train)  # , csids=csids_train)
    preds = clf.predict(X_test)  # , csids=csids_test)
    hits = misses = predictions = 0
    for pred, label in zip(preds, y_test):
        if pred == label:
            hits += 1
        else:
            misses += 1
        predictions += 1
    logging.info("Preds:" + str(predictions))
    logging.info("Hits:" + str(hits))
    logging.info("Misses:" + str(misses))
    return y_test, preds


if __name__ == '__main__':
    multiapp()
