import logging
import logging.config
import os
from pathlib import Path
import random
import tempfile
import yaml

import envoy
from sklearn.base import BaseEstimator
from tqdm import tqdm

from columbus.columbus import columbus


COLUMBUS_CACHE = Path('~/caches/columbus-cache').expanduser()


class Hybrid(BaseEstimator):
    """ scikit style class for hybrid method """
    def __init__(self, k=15, vw_binary='/home/ubuntu/bin/vw',
                 vw_args='-c --loss_function hinge -q :: --l2 0.005 '
                 '-b 25 --passes 300 --learning_rate 1.25 '
                 '--decay_learning_rate 0.95 --ftrl'):
        self.k = k
        self.vw_args = vw_args
        self.vw_binary = vw_binary
        self.indexed_labels = {}
        self.reverse_labels = {}
        # TODO: delete this file at __del__? after debugging
        self.vw_modelfile = tempfile.NamedTemporaryFile('w', delete=False)
        logging.info('Started hybrid model, vw_modelfile: %s',
                     self.vw_modelfile.name)

    def fit(self, X, y):
        logging.info('Training started')
        counter = 1
        for label in set(y):
            self.indexed_labels[label] = counter
            self.reverse_labels[counter] = label
            counter += 1
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
            '{vw_binary} {vw_input} {vw_args} '
            '--ect {ntags} -f {vw_modelfile}'.format(
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
                vw_modelfile=self.vw_modelfile.name)
        )
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. err: %s', c.std_err)
        os.unlink(f.name)
        return [self.reverse_labels[int(x)] for x in c.std_out.split()]

    def _columbize(self, X):
        logging.info('Getting columbus output for %d changesets', len(X))
        tags = []
        for changeset in tqdm(X):
            cshash = hash(tuple(sorted(changeset)))
            cache_file = COLUMBUS_CACHE / '{}.yaml'.format(cshash)
            if cache_file.exists():
                with cache_file.open('r') as f:
                    tags.append(yaml.load(f))
            else:
                tag = columbus(changeset, k=self.k)
                with cache_file.open('w') as f:
                    yaml.dump(tag, f)
                tags.append(tag)
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
