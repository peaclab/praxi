# Calculate F1 Scores for single label experiment
# Imports
from collections import Counter
from hashlib import md5
from multiprocessing import Lock
import os
from pathlib import Path
import random
import tempfile
import time
import yaml
import pickle
import copy

#import envoy
#from joblib import Memory
from sklearn.base import BaseEstimator
from tqdm import tqdm

import numpy as np
from numpy import savetxt
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from hybrid import Hybrid
from hybrid import Columbus
from rule_based import RuleBased

from columbus.columbus import columbus
from columbus.columbus import refresh_columbus


# Directory constants
PROJECT_ROOT = Path('~/hybrid-method').expanduser() # leave as project root to access any necessary files
CHANGESET_ROOT = Path('/home/ubuntu/caches/changesets/').expanduser()
COLUMBUS_CACHE = Path('/home/ubuntu/caches/columbus-cache-2').expanduser()
memory = Memory(cachedir='/home/ubuntu/caches/joblib-cache', verbose=0)
LABEL_DICT = Path('./pred_label_dict.pkl')

LOCK = Lock()


def gen_F1_scores(folder_path):
    # given the address to a folder with results and generate the F1 score for each file in said folder
    n_strats = 5
    filenames= []
    for root, dirs, files in os.walk("week2_results"):
        for filename in files:
            print(filename)
            filenames.append(filename)

    #y_true = [[] for _ in range(n_strats*3)]
    #y_pred = [[] for _ in range(n_strats*3)]
    #f1_scores = []

    #for i in range(len(filenames)):
    #    data = pickle.load(name)
    #    print(len(data)) # hopefully is 2
    #    y_true[i] = data[0]
    #    y_pred[i] = data[1]

    #labels = sorted(set(j for i in range(n_strats) for j in y_true[i]))

    #f1_weighted=[]

    #for x, y in zip(y_true, y_pred):
    #    f1_weighted.append(metrics.f1_score(x, y, labels, average = 'weighted'))

    # save results
    #resFile = 'F1_scores_single_label'
    # create a pickle file and dump
    #file = open(resFile, 'wb')
    #pickle.dump(filenames, file)
    #pickle.dump(f1_weighted, file)
    #file.close()

    #print("F1 Scores: ")
    #for i, j in zip(filenames, f1_weighted):
    #    print()
    #    print(i)

if __name__ == '__main__':
    fpath = '/home/ubuntu/praxi/single_label_results/preds'
    gen_F1_scores(fpath)
