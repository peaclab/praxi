import os
import logging
from pathlib import Path
import random
from random import randint
# from random import sample
import pdb

import yaml
from tqdm import tqdm

DATA_DIR = Path('/home/ubuntu/praxi/version_detection/changeset_compare/all')
#OUTPUT_DIR = Path('/projectnb/peaclab-mon/deltasherlock/results/')

def read_data(dirname, union=False, rate=1, threshold=1000,
                      exclude_app=''):
    """
    Read in data provided by Anthony.
    By default only 'union' files are used!!!
    Returns
    -------
    anthony_data : dict[str, dict[str, set[str]]]
        first dict contains a list of each label,
        second dict contains each yaml file parsed belonging to the label
        the set of strings contains the '$permissions $filename' strings
    """
    counter = dict()
    data = dict()
    filenames = os.listdir(dirname)
    for filename in tqdm(filenames):
        filename_label = filename.split('.')[0]
        if filename_label in counter and counter[filename_label] > threshold:
            continue
        if randint(0, 999)/1000.0 > rate:
            continue
        if 'yaml' not in filename:
            continue

        with open(os.path.join(dirname, filename), encoding='utf8') as f:
            filedata = yaml.load(f)
        if 'label' not in filedata:
            print(os.path.join(dirname, filename), " missed label !!!!")
            continue
        label = filedata['label']
        #label = ''.join([l for l in label if l.isalpha()])
        changes = set(filedata['changes'])
        #for c in changes:
        #    c = c[4:]
        if label not in data:
            data[label] = dict()
        data[label][filename] = changes
        if label not in counter:
            counter[label] = 0
        counter[label] += 1
    return data

def generate_rules(corpus):
    """Generates rules from the given corpus"""
    label_to_tokens = transform_anthony_intersection(corpus)
    # Filter out labels given by yum that refer to i686 architecture
    label_to_tokens = {k: v for k, v in label_to_tokens.items()
                       if k[-5:] != '.i686'}
    # Get the inverse map
    token_to_labels = get_token_to_labels(label_to_tokens)
    # Get the map from labels to categorized tokens
    label_to_token_groups = get_label_to_token_groups(token_to_labels)
    # Find duplicates
    duplicates = get_duplicates(label_to_tokens, token_to_labels,
                                label_to_token_groups)
    # Filter out duplicates from the corpus
    label_to_tokens = {k: v for k, v in label_to_tokens.items()
                       if k not in duplicates}
    # Again get the inverse map
    token_to_labels = get_token_to_labels(label_to_tokens)
    # Again get the map from labels to categorized tokens
    label_to_token_groups = get_label_to_token_groups(token_to_labels)
    # Generate rules for all labels
    rules = get_rules(label_to_tokens, token_to_labels,
                      label_to_token_groups, limit=1)
    logging.info('Finished rule generation')
    return rules


def transform_anthony_intersection(data):
    res = dict()
    for label in data:
        for filename in data[label]:
            if label not in res:
                res[label] = dict()
            for token in data[label][filename]:
                if token not in res[label]:
                    res[label][token] = 1
                else:
                    res[label][token] += 1
            # res[label] = res[label].intersection(data[label][filename])
    newres = dict()
    for label in res:
        newres[label] = set()
        maxval = max(res[label].values())
        for token in sorted(res[label], key=res[label].get, reverse=True):
            if res[label][token] != maxval and len(newres[label]) > 50:
                break
            if res[label][token] < 0.94 * maxval and len(newres[label]) >= 40:
                break
            if res[label][token] < 0.88 * maxval and len(newres[label]) >= 26:
                break
            if res[label][token] < 0.8 * maxval and len(newres[label]) >= 16:
                break
            if res[label][token] < 0.7 * maxval and len(newres[label]) >= 10:
                break
            if res[label][token] < 0.6 * maxval and len(newres[label]) >= 8:
                break
            if res[label][token] < 0.5 * maxval and len(newres[label]) >= 6:
                break
            newres[label].add(token)
    return newres

if __name__ == '__main__':
    prelim_data = read_data(DATA_DIR)
    print(len(prelim_data.keys()))

    applications = prelim_data.keys()

    for training_set_size, _ in enumerate(applications):
        for _ in range(5):
            training_set = random.sample(applications, training_set_size)
            rules = generate_rules( {app: files for app, files in prelim_data.items() if app in training_set})
            rules = {k: v for k, v in rules.items() if k in prelim_data.keys()}

            print(rules)
