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
from joblib import Memory
from sklearn.base import BaseEstimator
from tqdm import tqdm

from columbus.columbus import columbus


LOCK = Lock()
COLUMBUS_CACHE = Path('~/caches/columbus-cache').expanduser()
memory = Memory(cachedir='/home/centos/caches/joblib-cache', verbose=0)


class Hybrid(BaseEstimator):
    """ scikit style class for hybrid method """
    def __init__(self, freq_threshold=1, vw_binary='/home/centos/bin/vw',
                 pass_freq_to_vw=False,
                 vw_args='-c --stage_poly -b 26 --passes 1000 '
                 '--l1 1e-6 --l2 1e-6 --decay_learning_rate 0.995 '
                 '--ftrl',
                 probability=False, tqdm=True,
                 loss_function='hinge'):
        """ Initializer for Hybrid method. Do not use multiple instances
        simultaneously.
        """
        self.freq_threshold = freq_threshold
        self.vw_args = vw_args
        self.pass_freq_to_vw = pass_freq_to_vw
        self.probability = probability
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
        counter = 1
        all_labels = set()
        for labels in y:
            if isinstance(labels, list):
                for l in labels:
                    all_labels.add(l)
            else:
                all_labels.add(labels)
        for label in sorted(list(all_labels)):
            self.indexed_labels[label] = counter
            self.reverse_labels[counter] = label
            counter += 1
        if self.probability:
            self.vw_args_ += ' --csoaa {}'.format(len(all_labels))
        else:
            self.vw_args_ += ' --oaa {}'.format(len(all_labels))
            self.vw_args_ += ' --loss_function={}'.format(self.loss_function)
        tags = self._columbize(X)
        train_set = list(zip(tags, y))
        random.shuffle(train_set)
        f = tempfile.NamedTemporaryFile('w', delete=False)
        # f = open('./fit_input.txt', 'w')
        for tag, labels in train_set:
            if isinstance(labels, str):
                labels = [labels]
            input_string = ''
            for label, number in self.indexed_labels.items():
                if label in labels:
                    input_string += '{}:0.0 '.format(number)
                else:
                    input_string += '{}:1.0 '.format(number)
            f.write('{}| {}\n'.format(input_string, ' '.join(tag)))
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
        # f = open('./pred_input.txt', 'w')
        for tag in tags:
            f.write('{} | {}\n'.format(
                ' '.join([str(x) for x in self.reverse_labels.keys()]),
                ' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting testing', f.name)
        args = f.name
        if self.probability:
            args += ' -r /dev/stdout'
        else:
            args += ' -p /dev/stdout'
        c = envoy.run(
            '{vw_binary} {args} -t -i {vw_modelfile}'.format(
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
                if word == 'args':
                    break
                if word:
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
                tag = min(proba.keys(), key=lambda key: proba[key])
                proba.pop(tag)
                cur_top_k.append(self.reverse_labels[int(tag)])
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
            '{vw_binary} {vw_input} -t -p /dev/stdout -i {vw_modelfile}'.format(
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
        return _get_columbus_tags(X, disable_tqdm=(not self.tqdm),
                                  freq_threshold=self.freq_threshold,
                                  return_freq=self.pass_freq_to_vw)

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


class Columbus(BaseEstimator):
    """ scikit style class for columbus """
    def __init__(self, freq_threshold=2, tqdm=True):
        """ Initializer for columbus. Do not use multiple instances
        simultaneously.
        """
        self.freq_threshold = freq_threshold
        self.tqdm = tqdm

    def fit(self, X, y):
        pass

    def predict(self, X):
        tags = self._columbize(X)
        result = []
        for tagset in tags:
            result.append(max(tagset.keys(), key=lambda key: tagset[key]))
        return result

    def _columbize(self, X):
        mytags =  _get_columbus_tags(X, disable_tqdm=(not self.tqdm),
                                     freq_threshold=self.freq_threshold,
                                     return_freq=True)
        result = []
        for tagset in mytags:
            tagdict = {}
            for x in tagset:
                key, value = x.split(':')
                tagdict[key] = value
            result.append(tagdict)
        return result

@memory.cache
def _get_columbus_tags(X, disable_tqdm=False,
                       return_freq=True,
                       freq_threshold=2):
    logging.info('Getting columbus output for %d changesets', len(X))
    tags = []
    for changeset in tqdm(X, disable=disable_tqdm):
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
        if return_freq:
            tags.append(['{}:{}'.format(tag, freq) for tag, freq
                         in tag_dict.items()
                         if freq > freq_threshold])
        else:
            tags.append([tag for tag, freq in tag_dict.items()
                         if freq > freq_threshold])
    return tags
