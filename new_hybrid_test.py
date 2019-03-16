# Imports
from collections import Counter
from hashlib import md5
from multiprocessing import Lock

import os
from os import listdir
from os.path import isfile, join

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

from hybrid_tags import Hybrid

#from tagset_gen import Columbus

#from hybrid import Columbus
#from rule_based import RuleBased

from columbus.columbus import columbus
from columbus.columbus import refresh_columbus


# Directory constants
PROJECT_ROOT = Path('~/praxi').expanduser() # leave as project root to access any necessary files
#CHANGESET_ROOT = Path('/home/ubuntu/caches/changesets/').expanduser()
#COLUMBUS_CACHE = Path('/home/ubuntu/caches/columbus-cache-2').expanduser()
memory = Memory(cachedir='/home/ubuntu/caches/joblib-cache', verbose=0)
LABEL_DICT = Path('./pred_label_dict.pkl')

LOCK = Lock()

#######################################
#   FUNCTIONS for accessing tagsets   #
#######################################
def parse_ts(tagset_names, ts_dir):
    # Arguments: - tagset_names: a list of names of tagsets
    #            - ts_dir: the directory in which they are located
    # Returns: - tags: list of lists-- tags for each tagset name
    #          - labels: application name corresponding to each tagset
    tags = []
    labels = []
    for ts_name in tqdm(tagset_names):
            ts_path = ts_dir + '/' + ts_name
            tagset = get_tagset(ts_path)
            if 'labels' in tagset:
                # Multilabel changeset
                labels.append(tagset['labels'])
            else:
                labels.append(tagset['label'])
            tags.append(tagset['tags'])
    return tags, labels

def get_tagset(ts_path): # combine with parse_ts
    with open(ts_path, 'r') as stream:
        data_loaded = yaml.load(stream)
    return data_loaded

"""

def runtest(clean_csids, dirty_csids):
    # for now, just try one fold with 2000 dirty labels and 2500 clean labels
    # get the F1 score and the runtime
    # Weird constants you need
    suffix = 'hybrid'
    outdir = '/home/centos/hybrid-method/demo-results'
    iterative = False
    #####
    clean_train_csids = clean_csids[0] # just one list
    dirty_train_csids = dirty_csids[0] # again, just one list
    train_csids = clean_train_csids + dirty_train_csids
    X_train, y_train = parse_csids(train_csids) # X is the changesets, y is the corresponding package labels
    print(X_train[0])
    print(y_train[0])
    test_csids = dirty_csids[1] + dirty_csids[2]
    X_test, y_test = parse_csids(test_csids)
    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args='-b 26 --learning_rate 1.5',
                 suffix=suffix, iterative=iterative,
                 use_temp_files=True)
    # why is it called clf??? changeset labeling function... idk
    get_scores(clf, X_train, y_train, X_test, y_test)

def run_one_fold(train_csids, test_csids, filename):
    # Weird constants you need for ML model
    suffix = 'hybrid'
    outdir = '/home/centos/hybrid-method/demo-results'
    iterative = False
    ########################
    # run a fold given pre-partitioned CSID lists
    X_train, y_train = parse_csids(train_csids)
    #print(type(X_train))
    #print(type(X_train[0]))
    #print(X_train[0][0])
    X_test, y_test = parse_csids(test_csids)

    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args='-b 26 --learning_rate 1.5',
                 suffix=suffix, iterative=iterative,
                 use_temp_files=True)
    # Fit model
    print("Entering clf.fit(X_train, y_train)")
    clf.fit(X_train, y_train)
    # Get predictions
    y_preds = clf.predict(X_test) # will assign a label for each changeset
    # Now I have y_test and y_preds (save to file?)
    file = open(filename, "wb")
    pickle.dump(y_test, file)
    pickle.dump(y_preds, file)
    file.close()

    #results = zip(preds, y_test)
    #resultSet = set(results)
    #clf.score(X_test, y_test)
    #print(resultSet)
    #return F1score, runtime

def print_results(resfile, outdir, n_strats=5, args=None, iterative=False):
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


def runfolds(clean, dirty, resdir, part_file_names):
    # hard coded for now because I'm lazy but I'll fix it
    foldnum = ['0','1','2']
    for i in range(5):
        if i > 2: # already ran the others
            print("Loop #", i+1)
            clean_csids = []
            x = 0
            while x <= i-1:
                new_cleans = clean[x]
                print(len(new_cleans))
                x +=1
                clean_csids.extend(new_cleans)
            num_clean_sets = i
            for j in range(3):
                dirty_train_csids = dirty[j]
                dirty_test_csid_lists = dirty[:j] + dirty[j+1 :]
                dirty_test_csids = dirty_test_csid_lists[0] + dirty_test_csid_lists[1]
                train_csids = clean_csids + dirty_train_csids
                # Create name of file
                fname = resdir + part_file_names[i] + foldnum[j] + ".p"
                print(fname)
                # RUN THE FOLD
                run_one_fold(train_csids, dirty_test_csids, fname)
                #F1score, runtime = run_one_fold(train_csids, dirty_test_csids)

def gen_F1_scores(folder_path):
    # given the address to a folder with results and generate the F1 score for each file in said folder
    n_strats = 5
    filenames= []
    for root, dirs, files in os.walk("week2_results"):
        for filename in files:
            print(filename)
            filenames.append(filename)

    y_true = [[] for _ in range(n_strats*3)]
    y_pred = [[] for _ in range(n_strats*3)]

    for i in range(len(filenames)):
        curFileName = folder_path + filenames[i]
        file = open(curFileName, 'rb')
        y_true[i] = pickle.load(file)
        y_pred[i] = pickle.load(file)
        print(len(y_true[i]), len(y_pred[i])) # hopefully is 2000
        file.close()

    labels = sorted(set(j for i in range(n_strats) for j in y_true[i]))

    f1_weighted=[]

    for x, y in zip(y_true, y_pred):
        print("New Loop:")
        print(len(x))
        print(len(y))
        print(len(labels))
        f1_weighted.append(metrics.f1_score(x, y, labels, average = 'weighted'))

    # save results
    resFile = 'week3_results/F1_scores_single_label.p'
    # create a pickle file and dump
    file = open(resFile, 'wb')
    pickle.dump(filenames, file)
    pickle.dump(f1_weighted, file)
    file.close()

    print("F1 Scores: ")
    for i, j in zip(filenames, f1_weighted):
        print(i)
        print(j)

def mean(list):
    sum = 0
    for i in list:
        sum += i
    return sum/len(list)


def avg_F1(fileName):
    # I have a list of 15 F1 scores, need to average every 3 to get the
    # official F1 for every experiment
    file = open(fileName, 'rb')
    fnames = pickle.load(file)
    scores = pickle.load(file)
    print(len(fnames), len(scores)) # 15, good
    file.close()

    for i in scores:
        print(i)

    exp_names = ["0 Clean", "2500 Clean", "5000 Clean", "7500 Clean", "10000 Clean"]
    f1_scores = [mean(scores[0:3]),mean(scores[3:6]),mean(scores[6:9]),mean(scores[9:12]),mean(scores[12:15])]
    for i, j in zip(exp_names, f1_scores):
        print(i, j)

    resFile = 'week3_results/F1_scores_single_label_FINAL.p'
    # create a pickle file and dump
    file = open(resFile, 'wb')
    pickle.dump(exp_names, file)
    pickle.dump(f1_scores, file)
    file.close()

def dataTest(file_path):
    file = open(file_path, 'rb')
    y_test = pickle.load(file)
    y_preds = pickle.load(file)
    for i in range(len(y_test)):
        if(y_test[i] != y_preds[i]):
            print(y_test[i], y_preds[i], "WRONG")
        else:
            print(y_test[i], y_preds[i])
    #print(len(data))
    #print(data) # they JUST contain the guesses I think. So I will have to rerun
"""

if __name__ == '__main__':
    ts_path = '/home/ubuntu/praxi/week5/multitest_tags'
    ts_names = [f for f in listdir(ts_path) if isfile(join(ts_path, f))]
    # ^ Make sure to filter specifically for tagsets

    tags, labels = parse_ts(ts_names, ts_path)
    # ^^ This is what I will pass into the clf.fit function


    suffix = 'hybrid'
    iterative = False
    # see if fit runs w/o error... CHECK
    clf = Hybrid(freq_threshold=2, pass_freq_to_vw=True, probability=False,
                 vw_args='-b 26 --learning_rate 1.5',
                 suffix=suffix, iterative=iterative,
                 use_temp_files=True)


    all_preds = clf.predict(tags)

    # Fit model
    print("Entering clf.fit(X_train, y_train)")
    clf.fit(tags, labels)

    # Now try to use predict! :) ... need a test set...


    # TEST FIT

    #print("Num tags: ", len(tags))
    #print("Labels: ", labels)

    #runtest(clean, dirty)
    #runfolds(clean, dirty, res_dir, part_file_names)
    #run_one_fold(dirty[0][0:5], dirty[1][0:5], "results/firstTest.p")
    print("Done!")
