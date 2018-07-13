#!/usr/bin/python
# -*- coding: utf-8 -*-

import yaml
import os
import random


def merge_changeset_yaml(list_of_changesets, save_path=False):
    """
    Synthesizes a multi-label changeset from a provided list of changeset
    YAML files and optionally saves the changeset to disk.

    :param list_of_changesets: any number of paths to changeset YAML files (in string form)
    :param save_path: Set to a string path to save the multilabel changeset to a YAML file.
       Set to False to disable saving (default)
    :returns: a multilabel changeset in dictionary format
    """

    # Setup what will become the multi-label changeset (use arbitrarily large values for open and close times)
    # Remember that multi-label changesets have a field named 'labels' while single-label ones only have 'label'
    result = {
        'labels': [],
        'open_time': 32503680000,
        'close_time': -32503680000,
        'changes': [],
        }
    for cs_path in list_of_changesets:
        with open(cs_path, 'r') as f:

            # Load YAML file into a dict
            single = yaml.load(f)

            # Copy label
            result['labels'].append(single['label'])

            # Copy open and close times
            result['open_time'] = (single['open_time'] if single['open_time'] < result['open_time'] else result['open_time'])
            result['close_time'] = (single['close_time'] if single['close_time'] > result['close_time'] else result['close_time'])

            # Copy changes
            result['changes'] = result['changes'] + single['changes']

    # Save yaml file if desired
    if save_path != False:
        with open(save_path, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)

    # Return dictionary object
    return result


def gen_pairings(source_dir, num_apps=2, qty=100):
    """
    Generates random doubles (or triples, quadruples, etc.) of single-app changesets that
    can be merged together with merge_changeset_yaml(). Only returns filenames, does not
    do the actual merging.

    :param source_dir: the source directory containing the changeset YAML files (naming convention must be followed)
    :param num_apps: the number of changesets per pairing. Default: 2
    :param qty: the number of pairings to generate
    :returns: a list of pairings of filenames (in tuple format)
    """

    # Standardize the source directory path
    source_dir = os.path.abspath(source_dir)

    # Get and filter dir listing
    listing = [os.path.join(source_dir, x) for x in
               os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, x))
               and len(x.split('.')) == 6 and x.split('.')[5] == 'yaml']
    dirty = [x for x in listing
             if x.split('.')[4] == 'ts' and '.vd.' not in x and 'apache2' not in x]
    print("Discovered {} usable changeset files, {} of which are dirty".format(len(listing), len(dirty)))

    # Seed the RNG
    random.seed()

    # Generate pairings
    pairings = []
    for i in range(qty):
        pair = []

        # Keep getting random samples until there are no repeats of apps in the sample
        while len(set([os.path.basename(x).split('.')[0] for x in pair])) < num_apps:
            pair = random.sample(dirty, num_apps)

        # Record the pairing
        pairings.append(pair)

    return pairings

####################################################
# DEMO driver code (modify this to fit your setup) #
####################################################
from pickle import dump
SOURCE_DIR = "/home/centos/caches/changesets"
OUTPUT_DIR = "/home/centos/caches/multi-app-changesets"
NUM_APPS_MIN = 2
NUM_APPS_MAX = 5
QTY = 200

for n in range(NUM_APPS_MIN, NUM_APPS_MAX):
    # Generate pairings
    pairings = gen_pairings(SOURCE_DIR, n, QTY)

    # PLEASE SAVE PAIRINGS so we can re-use them for future experiments
    with open(str(n) + "_app_pairings.p", 'wb') as f:
        dump(pairings, f)

    # Merge changesets and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, pair in enumerate(pairings):
        merge_changeset_yaml(pair, os.path.join(OUTPUT_DIR, "{}_app_{}.yaml".format(n, i)))

    print("Successfully synthesized {} {}-app changesets".format(QTY, n))

