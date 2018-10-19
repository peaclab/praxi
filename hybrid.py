from collections import Counter
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
COLUMBUS_CACHE = Path('~/caches/columbus-cache-2').expanduser()
memory = Memory(cachedir='/home/centos/caches/joblib-cache', verbose=0)


class Hybrid(BaseEstimator):
    """ scikit style class for hybrid method """
    def __init__(self, freq_threshold=1, vw_binary='/home/centos/bin/vw',
                 pass_freq_to_vw=False, pass_files_to_vw=False,
                 vw_args='-b 26 --passes=20 -l 50',
                 probability=False, tqdm=True,
                 suffix='', iterative=False,
                 loss_function='hinge',
                 use_temp_files=False):
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
        self.pass_files_to_vw = pass_files_to_vw
        self.suffix = suffix
        self.iterative = iterative
        self.use_temp_files = (not self.iterative) and use_temp_files

    def get_args(self):
        return self.vw_args_

    def refresh(self):
        """Remove all cached files, reset iterative training."""
        if hasattr(self, 'vw_modelfile'):
            os.unlink(self.vw_modelfile)
            self.indexed_labels = {}
            self.reverse_labels = {}
            self.all_labels = set()
            self.label_counter = 1

    def fit(self, X, y):
        if not self.probability:
            X, y = self._filter_multilabels(X, y)
        if self.use_temp_files:
            modelfileobj = tempfile.NamedTemporaryFile('w', delete=False)
            self.vw_modelfile = modelfileobj.name
            modelfileobj.close()
        else:
            self.vw_modelfile = 'trained_model-%s.vw' % self.suffix
            if not (self.iterative and hasattr(self, 'indexed_labels')):
                try:
                    os.unlink(self.vw_modelfile)
                except FileNotFoundError:
                    pass
            else:
                logging.info("Using old vw_modelfile: %s", self.vw_modelfile)
        logging.info('Started hybrid model, vw_modelfile: %s',
                     self.vw_modelfile)
        self.vw_args_ = self.vw_args
        if not (self.iterative and hasattr(self, 'indexed_labels')):
            self.indexed_labels = {}
            self.reverse_labels = {}
            self.all_labels = set()
            self.label_counter = 1
        for labels in y:
            if isinstance(labels, list):
                for l in labels:
                    self.all_labels.add(l)
            else:
                self.all_labels.add(labels)
        for label in sorted(list(self.all_labels)):
            if label not in self.indexed_labels:
                self.indexed_labels[label] = self.label_counter
                self.reverse_labels[self.label_counter] = label
                self.label_counter += 1
        if self.probability:
            self.vw_args_ += ' --csoaa {}'.format(len(self.all_labels))
        else:
            self.vw_args_ += ' --probabilities'
            self.loss_function = 'logistic'
            self.vw_args_ += ' --loss_function={}'.format(self.loss_function)
            if self.iterative:
                self.vw_args_ += ' --oaa 80'
            else:
                self.vw_args_ += ' --oaa {}'.format(len(self.all_labels))
        if self.iterative:
            self.vw_args_ += ' --save_resume'
        self.vw_args_ += ' --kill_cache --cache_file a.cache'
        tags = self._get_tags(X)
        train_set = list(zip(tags, y))
        random.shuffle(train_set)
        if self.use_temp_files:
            f = tempfile.NamedTemporaryFile('w', delete=False)
        else:
            with open('./label_table-%s.yaml' % self.suffix, 'w') as f:
                yaml.dump(self.reverse_labels, f)
            f = open('./fit_input-%s.txt' % self.suffix, 'w')
        for tag, labels in train_set:
            if isinstance(labels, str):
                labels = [labels]
            input_string = ''
            if self.probability:
                for label, number in self.indexed_labels.items():
                    if label in labels:
                        input_string += '{}:0.0 '.format(number)
                    else:
                        input_string += '{}:1.0 '.format(number)
            else:
                input_string += '{} '.format(self.indexed_labels[labels[0]])
            f.write('{}| {}\n'.format(input_string, ' '.join(tag)))
        f.close()
        command = '{vw_binary} {vw_input} {vw_args} -f {vw_modelfile}'.format(
            vw_binary=self.vw_binary, vw_input=f.name,
            vw_args=self.vw_args_, vw_modelfile=self.vw_modelfile)
        logging.info('vw input written to %s, starting training', f.name)
        logging.info('vw command: %s', command)
        c = envoy.run(command)
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. out: %s, err: %s',
                c.std_out, c.std_err)
        if self.use_temp_files:
            os.unlink(f.name)

    def predict_proba(self, X):
        tags = self._get_tags(X)
        if self.use_temp_files:
            f = tempfile.NamedTemporaryFile('w', delete=False)
            outfobj = tempfile.NamedTemporaryFile('w', delete=False)
            outf = outfobj.name
            outf.close()
        else:
            f = open('./pred_input-%s.txt' % self.suffix, 'w')
            outf = './pred_output-%s.txt' % self.suffix
        if self.probability:
            for tag in tags:
                f.write('{} | {}\n'.format(
                    ' '.join([str(x) for x in self.reverse_labels.keys()]),
                    ' '.join(tag)))
        else:
            for tag in tags:
                f.write('| {}\n'.format(' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting testing', f.name)
        args = f.name
        args += ' -r %s' % outf
        command = '{vw_binary} {args} -t -i {vw_modelfile}'.format(
            vw_binary=self.vw_binary, args=args,
            vw_modelfile=self.vw_modelfile)
        logging.info('vw command: %s', command)
        c = envoy.run(command)
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. out: %s, err: %s',
                c.std_out, c.std_err)
        all_probas = []
        with open(outf, 'r') as f:
            for line in f:
                probas = {}
                for word in line.split(' '):
                    tag, p = word.split(':')
                    probas[tag] = float(p)
                all_probas.append(probas)
        if self.use_temp_files:
            os.unlink(f.name)
            os.unlink(self.vw_modelfile)
            os.unlink(outf)
        return all_probas

    def top_k_tags(self, X, ntags):
        probas = self.predict_proba(X)
        result = []
        for ntag, proba in zip(ntags, probas):
            cur_top_k = []
            for i in range(ntag):
                if self.probability:
                    tag = min(proba.keys(), key=lambda key: proba[key])
                else:
                    tag = max(proba.keys(), key=lambda key: proba[key])
                proba.pop(tag)
                cur_top_k.append(self.reverse_labels[int(tag)])
            result.append(cur_top_k)
        return result

    def predict(self, X):
        tags = self._get_tags(X)
        if self.use_temp_files:
            f = tempfile.NamedTemporaryFile('w', delete=False)
            outfobj = tempfile.NamedTemporaryFile('w', delete=False)
            outf = outfobj.name
            outf.close()
        else:
            f = open('./pred_input-%s.txt' % self.suffix, 'w')
            outf = './pred_output-%s.txt' % self.suffix
        for tag in tags:
            f.write('| {}\n'.format(' '.join(tag)))
        f.close()
        logging.info('vw input written to %s, starting testing', f.name)
        command = '{vw_binary} {vw_input} -t -p {outf} -i {vw_modelfile}'.format(
            vw_binary=self.vw_binary, vw_input=f.name, outf=outf,
            vw_modelfile=self.vw_modelfile)
        logging.info('vw command: %s', command)
        c = envoy.run(command)
        if c.status_code:
            logging.error(
                'something happened to vw, code: %d, out: %s, err: %s',
                c.status_code, c.std_out, c.std_err)
            raise IOError('Something happened to vw')
        else:
            logging.info(
                'vw ran sucessfully. out: %s, err: %s',
                c.std_out, c.std_err)
        all_preds = []
        with open(outf, 'r') as f:
            for line in f:
                all_preds.append(int(line))
        if self.use_temp_files:
            os.unlink(f.name)
            os.unlink(self.vw_modelfile)
        return [self.reverse_labels[x] for x in all_preds]

    def _get_tags(self, X):
        if self.pass_files_to_vw:
            return _get_filename_frequencies(X, disable_tqdm=(not self.tqdm),
                                             freq_threshold=self.freq_threshold)
        return _get_columbus_tags(X, disable_tqdm=(not self.tqdm),
                                  freq_threshold=self.freq_threshold,
                                  return_freq=self.pass_freq_to_vw)

    def _filter_multilabels(self, X, y):
        new_X = []
        new_y = []
        for data, labels in zip(X, y):
            if isinstance(labels, list) and len(labels) == 1:
                new_X.append(data)
                new_y.append(labels[0])
            elif isinstance(labels, str):
                new_X.append(data)
                new_y.append(labels)
        return new_X, new_y

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
def _get_filename_frequencies(X, disable_tqdm=False, freq_threshold=2):
    logging.info("Getting filename frequencies for %d changesets", len(X))
    tags = []
    for changeset in tqdm(X, disable=disable_tqdm):
        c = Counter()
        for filename in changeset:
            c.update(filename.split(' ')[1].split('/'))
        del c['']
        tags.append(['{}:{}'.format(tag.replace(':', '').replace('|', ''), freq)
                     for tag, freq in dict(c).items() if freq > freq_threshold])
    return tags

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
