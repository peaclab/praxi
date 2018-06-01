#!/usr/bin/env python3

import logging
import logging.config
import multiprocessing
import os
import pickle
from pathlib import Path
import tempfile
import time
import yaml

import envoy
from tqdm import tqdm
from numpy import savetxt
from sklearn import metrics
from joblib import Memory

from columbus.columbus import columbus

PROJECT_ROOT = Path('~/hybrid-method').expanduser()
CHANGESET_ROOT = Path('~/yaml/').expanduser()
COLUMBUS_CACHE = Path('~/columbus-cache').expanduser()
memory = Memory(cachedir='/home/ubuntu/joblib-cache', verbose=0)


def main():
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
    # get_scores([101289, 102580, 102585, 99234],
    #            [101289, 102580, 102585, 99234])
    with (PROJECT_ROOT / 'changeset_sets' /
          'threek_dirty_chunks.p').open('rb') as f:
        threeks = pickle.load(f)
    with (PROJECT_ROOT / 'changeset_sets' /
          'tenk_clean_chunks.p').open('rb') as f:
        tenks = pickle.load(f)
    # resfile = open('./results.pkl', 'wb')
    results = []
    for idx, test_csids in tqdm(enumerate(threeks)):
        logging.info('Test set is %d', idx)
        train_idx = [0, 1, 2]
        train_idx.remove(idx)
        # # Split calls to parse_csids for more efficient memoization
        # X_train, y_train = parse_csids(threeks[train_idx[0]])
        # features, labels = parse_csids(threeks[train_idx[1]])
        # X_train += features
        # y_train += labels
        # X_test, y_test = parse_csids(threeks[idx])
        # train_csids = threeks[train_idx[0]] + threeks[train_idx[1]]
        # results.append(get_scores(X_train, y_train, train_csids,
        #                           X_test, y_test, test_csids))
        # pickle.dump(results, resfile)
        # resfile.seek(0)
        for inner_idx, extra_cleans in tqdm(enumerate(tenks)):
            logging.info('Extra clean count: %d', inner_idx + 1)
            features, labels = parse_csids(extra_cleans)
            # X_train += features
            # y_train += labels
            # train_csids += extra_cleans
            # results.append(get_scores(X_train, y_train, train_csids,
            #                           X_test, y_test, test_csids))
            # pickle.dump(results, resfile)
            # resfile.seek(0)
    # resfile.close()
    # # Now do the evaluation!
    # #results = [
    # #    0 => ([x, y, z], <-- true
    # #          [x, y, k]) <-- pred
    # #]
    y_true = [[], [], [], [], []]
    y_pred = [[], [], [], [], []]
    for idx, result in enumerate(results):
        y_true[idx % 5] += result[0]
        y_pred[idx % 5] += result[1]

    labels = sorted(set(y_true[0] + y_true[1] + y_true[2] +
                        y_true[3] + y_true[4]))
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
            range(5), classifications, f1_weighted, f1_micro, f1_macro,
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
        savetxt("/home/ubuntu/hybrid-results/{}.txt".format(strat),
                confuse, fmt='%d', header=file_header, delimiter=',',
                comments='')


class Hybrid:
    """ scikit style class for hybrid method """
    def __init__(self, k=15, vw_binary='/home/ubuntu/bin/vw',
                 vw_args='-c --loss_function hinge -q cc '
                         '-b 25 --passes 50 -l 0.4'):
        self.k = k
        self.vw_args = vw_args
        self.vw_binary = vw_binary
        self.indexed_labels = {}
        self.reverse_labels = {}
        # TODO: delete this file at __del__? after debugging
        self.vw_modelfile = tempfile.NamedTemporaryFile('w', delete=False)
        logging.info('Started hybrid model, vw_modelfile: %s',
                     self.vw_modelfile.name)

    def fit(self, X, y, csids=None):
        logging.info('Training started')
        counter = 1
        for label in set(y):
            self.indexed_labels[label] = counter
            self.reverse_labels[counter] = label
            counter += 1
        tags = self._columbize(X, csids=csids)
        f = tempfile.NamedTemporaryFile('w', delete=False)
        for tag, label in zip(tags, y):
            f.write('{} | {}\n'.format(
                self.indexed_labels[label],
                ' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting training', f.name)
        c = envoy.run(
            '{vw_binary} {vw_input} {vw_args} '
            '--oaa {ntags} -f {vw_modelfile}'.format(
                vw_binary=self.vw_binary, vw_input=f.name, ntags=len(set(y)),
                vw_args=self.vw_args, vw_modelfile=self.vw_modelfile.name)
        )
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. out: %s, err: %s',
                c.std_out, c.std_err)
        os.unlink(f.name)

    def predict(self, X, csids=None):
        tags = self._columbize(X, csids=csids)
        f = tempfile.NamedTemporaryFile('w', delete=False)
        for tag in tags:
            f.write('| {}\n'.format(' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting testing', f.name)
        c = envoy.run(
            '{vw_binary} {vw_input} -p /dev/stdout -i {vw_modelfile}'.format(
                vw_binary=self.vw_binary, vw_input=f.name,
                vw_modelfile=self.vw_modelfile.name)
        )
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. out: %s, err: %s',
                c.std_out, c.std_err)
        os.unlink(f.name)
        return [self.reverse_labels[int(x)] for x in c.std_out.split()]

    def _columbize(self, X, csids=None):
        logging.info('Getting columbus output for %d changesets', len(X))
        if csids is None:
            csids = [-1 for _ in X]
        tags = []
        for changeset, csid in tqdm(zip(X, csids)):
            if csid != -1:
                cache_file = COLUMBUS_CACHE / '{}.yaml'.format(csid)
                if cache_file.exists():
                    with cache_file.open('r') as f:
                        tags.append(yaml.load(f))
                else:
                    tag = columbus(changeset, k=self.k)
                    with cache_file.open('w') as f:
                        yaml.dump(tag, f)
                    tags.append(tag)
            else:
                tags.append(columbus(changeset, k=self.k))
        return tags

    def score(self, X, y, csids=None):
        predictions = self.predict(X, csids=csids)
        logging.info('Getting scores')
        hits = misses = preds = 0
        for pred, label in zip(predictions, y):
            if int(self.indexed_labels[label]) == int(pred):
                hits += 1
            else:
                misses += 1
            preds += 1
        print("Preds:" + str(preds))
        print("Hits:" + str(hits))
        print("Misses:" + str(misses))
        return {'preds': preds, 'hits': hits, 'misses': misses}


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
def parse_csids(csids):
    """ Returns labels and features from csids, features are file sets
    file sets: list of string of format '644 /usr/.../file' """
    features = []
    labels = []
    for csid in tqdm(csids):
        changeset = get_changeset(csid)
        labels.append(changeset['label'])
        features.append(changeset['changes'])
    return features, labels


def get_scores(X_train, y_train, csids_train, X_test, y_test, csids_test):
    """ Gets two lists of changeset ids, does training+testing """
    clf = Hybrid()
    clf.fit(X_train, y_train, csids=csids_train)
    preds = clf.predict(X_test, csids=csids_test)
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
    main()
