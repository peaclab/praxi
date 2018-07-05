import logging
import logging.config
from hashlib import md5
from multiprocessing import Lock
import os
from pathlib import Path
import random
import tempfile
import yaml

import envoy
import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from columbus.columbus import columbus


LOCK = Lock()
COLUMBUS_CACHE = Path('~/caches/columbus-cache').expanduser()


class Hybrid(BaseEstimator):
    """ scikit style class for hybrid method """
    def __init__(self, freq_threshold=1, vw_binary='/home/centos/bin/vw',
                 pass_freq_to_vw=False,
                 vw_args='-c -q :: --l2 0.005 -b 25 --passes 300 '
                 '--learning_rate 1.25 --decay_learning_rate 0.95 --ftrl',
                 probability=False, tqdm=True,
                 probability_args=' --link=logistic',
                 loss_function='hinge'):
        """ Initializer for Hybrid method. Do not use multiple instances
        simultaneously.
        """
        self.freq_threshold = freq_threshold
        self.vw_args = vw_args
        self.pass_freq_to_vw = pass_freq_to_vw
        self.probability = probability
        self.probability_args = probability_args
        self.loss_function = loss_function
        self.vw_binary = vw_binary
        self.tqdm = tqdm

    def fit(self, X, y):
        modelfileobj = tempfile.NamedTemporaryFile('w', delete=False)
        self.vw_modelfile = modelfileobj.name
        modelfileobj.close()
        logging.info('Started hybrid model, vw_modelfile: %s',
                     self.vw_modelfile)
        self.vw_args_ = self.vw_args
        self.indexed_labels = {}
        self.reverse_labels = {}
        if self.probability:
            self.loss_function = 'logistic'
            if len(set(y)) > 2:
                self.vw_args_ += ' --oaa {}'.format(len(set(y)))
                self.vw_args_ += ' --probabilities'
                counter = 1
                for label in set(y):
                    self.indexed_labels[label] = counter
                    self.reverse_labels[counter] = label
                    counter += 1
            else:
                self.vw_args_ += self.probability_args
                self.indexed_labels = {1: 1, 0: -1}
                self.reverse_labels = {1: 1, -1: 0}
        else:
            self.vw_args_ += ' --oaa {}'.format(len(set(y)))
            counter = 1
            for label in set(y):
                self.indexed_labels[label] = counter
                self.reverse_labels[counter] = label
                counter += 1
        self.vw_args_ += ' --loss_function={}'.format(self.loss_function)
        tags = self._columbize(X)
        train_set = list(zip(tags, y))
        random.shuffle(train_set)
        f = tempfile.NamedTemporaryFile('w', delete=False)
        for tag, label in train_set:
            f.write('{} | {}\n'.format(
                self.indexed_labels[label],
                ' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting training', f.name)
        c = envoy.run(
            '{vw_binary} {vw_input} {vw_args} -f {vw_modelfile}'.format(
                vw_binary=self.vw_binary, vw_input=f.name,
                vw_args=self.vw_args_, vw_modelfile=self.vw_modelfile)
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

    def predict_proba(self, X):
        tags = self._columbize(X)
        f = tempfile.NamedTemporaryFile('w', delete=False)
        for tag in tags:
            f.write('| {}\n'.format(' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting testing', f.name)
        args = f.name
        if self.probability:
            # self.loss_function = 'logistic'
            # args += self.probability_args
            # args += ' --loss_function={}'.format(self.loss_function)
            args += ' -r /dev/stdout'
        else:
            args += ' -p /dev/stdout'
        c = envoy.run(
            '{vw_binary} {args} -i {vw_modelfile}'.format(
                vw_binary=self.vw_binary, args=args,
                vw_modelfile=self.vw_modelfile)
        )
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. one prediction: %s, err: %s',
                c.std_out.split()[0], c.std_err)
        os.unlink(f.name)
        os.unlink(self.vw_modelfile)
        all_probas = []
        for line in c.std_out.split('\n'):
            probas = {}
            for word in line.split(' '):
                tag, p = word.split(':')
                probas[tag] = float(p)
            all_probas.append(probas)
        return all_probas

    def top_k_tags(self, X, ntags):
        probas = self.predict_proba(X)
        result = []
        for ntag, proba in zip(ntags, probas):
            cur_top_k = []
            for i in range(ntag):
                tag = max(proba.keys(), key=lambda key: proba[key])
                cur_top_k.append(tag)
                proba.pop(tag)
            result.append(cur_top_k)
        return result

    def predict(self, X):
        tags = self._columbize(X)
        f = tempfile.NamedTemporaryFile('w', delete=False)
        for tag in tags:
            f.write('| {}\n'.format(' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting testing', f.name)
        c = envoy.run(
            '{vw_binary} {vw_input} -p /dev/stdout -i {vw_modelfile}'.format(
                vw_binary=self.vw_binary, vw_input=f.name,
                vw_modelfile=self.vw_modelfile)
        )
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. one prediction: %s, err: %s',
                c.std_out.split()[0], c.std_err)
        os.unlink(f.name)
        os.unlink(self.vw_modelfile)
        return [self.reverse_labels[int(x)] for x in c.std_out.split()]

    def _columbize(self, X):
        logging.info('Getting columbus output for %d changesets', len(X))
        tags = []
        for changeset in tqdm(X, disable=(not self.tqdm)):
            cshash = md5(str(sorted(changeset)).encode()).hexdigest()
            cache_file = COLUMBUS_CACHE / '{}.yaml'.format(cshash)
            if cache_file.exists():
                with cache_file.open('r') as f:
                    tag_dict = yaml.load(f)
            else:
                with LOCK:
                    tag_dict = columbus(changeset)
                with cache_file.open('w') as f:
                    yaml.dump(tag_dict, f)
            if self.pass_freq_to_vw:
                tags.append(['{}:{}'.format(tag, freq) for tag, freq
                             in tag_dict.items()
                             if freq > self.freq_threshold])
            else:
                tags.append([tag for tag, freq in tag_dict.items()
                             if freq > self.freq_threshold])
        return tags

    def score(self, X, y):
        predictions = self.predict(X)
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
