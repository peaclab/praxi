#!/usr/bin/python3
# Script function:
#      - given a directory of changesets, create a directory containing the corresponding tagsets
# COMMAND LINE INPUTS:
#      - one input: changeset directory
#      - two inputs: changset directory, tagset directory (IN THAT ORDER)
#                    * changeset directory must exist, if tagset directory does not exist it will be created
############################ TEST THIS FILE ########################################
# All functionalities seem to be working, so now I will try to delete some dead code

# Imports
from collections import Counter
from multiprocessing import Lock

import os
from os import listdir
from os.path import isfile, join, isabs
import sys

from pathlib import Path
import random
import time
import yaml

import envoy
from joblib import Memory
#from sklearn.base import BaseEstimator
from tqdm import tqdm

from columbus.columbus import columbus
from columbus.columbus import refresh_columbus

#  Change for use on local machine
LOCK = Lock()
#CHANGESET_ROOT = Path('~/caches/changesets/').expanduser()
#COLUMBUS_CACHE = Path('/home/ubuntu/caches/columbus-cache-2').expanduser()
memory = Memory(cachedir='/home/ubuntu/caches/joblib-cache', verbose=0)


@memory.cache # what does this do...?
def parse_cs(changeset_names, cs_dir, multilabel=False, iterative=False):
    # input: list of changeset names (strings)
    # output: a list of labels and a corresponding list of features (list of filepaths of changed/added files)
    features = []
    labels = []
    for cs_name in tqdm(changeset_names):
            changeset = get_changeset(cs_name, cs_dir, iterative=iterative)
            if multilabel:
                if 'labels' in changeset:
                    labels.append(changeset['labels'])
                else:
                    labels.append(changeset['label'])
            else:
                labels.append(changeset['label'])
            features.append(changeset['changes'])
    return features, labels

def get_changeset(cs_fname, cs_dir, iterative=False):
    # input: file name of a *single* changeset
    # output: dictionary containing changed/added filepaths and label(s)
    cs_dir_obj = Path(cs_dir).expanduser()
    changeset = None
    for csfile in cs_dir_obj.glob(cs_fname): # CHANGE THIS
        if changeset is not None:
            raise IOError("Too many changesets match the file name")
        with csfile.open('r') as f:
            changeset = yaml.load(f)
    if changeset is None:
        raise IOError("No changesets match the name")
    if 'changes' not in changeset or ('label' not in changeset and 'labels' not in changeset):
        raise IOError("Couldn't read changeset")
    return changeset

def _get_columbus_tags(X, disable_tqdm=False,
                       return_freq=True,
                       freq_threshold=2):
    # Returns a list of tags and their frequency (as strings)
    tags = []
    for changeset in tqdm(X, disable=disable_tqdm):
        #print("changeset data type", type(changeset))
        tag_dict = columbus(changeset, freq_threshold=freq_threshold)
        if return_freq:
            tags.append(['{}:{}'.format(tag, freq) for tag, freq
                         in tag_dict.items()])
        else:
            tags.append([tag for tag, freq in tag_dict.items()])
    return tags

def create_tagset_names(changeset_names):
    # input: list of changeset names
    # output: list of names for tagsets created for these changesets
    tagset_names = []
    for name in changeset_names:
        new_tagname = name[:-4] + "tag"
        #print(new_tagname)
        tagset_names.append(new_tagname)
    #print('TAGSET_NAMES ELEM', tagset_names[0])
    return tagset_names

def get_changeset_names(cs_dir):
    # takes a directory name and returns all changeset files within the directory
    all_files = [f for f in listdir(cs_dir) if isfile(join(cs_dir, f))]
    changeset_names = [f for f in all_files if ".yaml" in f and ".tag" not in f]
    return changeset_names

def get_ids(changeset_names):
    # return the ids of all changesets in a list (between first and second '.')
    c_ids = []
    c = '.'
    for cs_name in changeset_names:
        idxs = [pos for pos, char in enumerate(cs_name) if char == c]
        curr_id = cs_name[idxs[0]+1:idxs[1]]
        c_ids.append(curr_id)
    return c_ids

def create_files(tagset_names, ts_dir, labels, ids, tags):
    # add tags, labels, ids to yaml files
    for i, tagset_name in enumerate(tagset_names):
        #if()
        # CHANGE "labels" to "label" if not multi?
        # multilabel changeset
        if(isinstance(labels[i], list)):
            cur_dict = {'labels': labels[i], 'id' : ids[i], 'tags': tags[i]}
            print("multi!")
        else:
            cur_dict = {'label': labels[i], 'id' : ids[i], 'tags': tags[i]}
        #print(tagset_names[i])
        cur_fname = ts_dir + '/' + tagset_name
        with open(cur_fname, 'w') as outfile:
            yaml.dump(cur_dict, outfile, default_flow_style=False)


def get_free_filename(stub, directory, suffix=''):
    counter = 0
    while True:
        file_candidate = '{}/{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            counter += 1
        else:  # No match found
            if suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate

def create_res_dir(work_dir, path_str=""): # THIS IS DONE I THINK
    # Given a path string, check if it exists, if is doesn't create a directory
    # return full file path
    if (path_str!=""): # path specified
        if not os.path.isabs(path_str):
            # IF PATH IS NOT ABSOLUTE, ASSUMED TO BE RELATIVE
            path_str = work_dir + '/' + path_str
        # check if directory exists, if it doesn't, create one!
        if not os.path.isdir(path_str):
            os.mkdir(path_str)
    else: # no path specified
        # create a directory in the working directory for tagsets
        path_str = get_free_filename('tagsets', work_dir)
    return path_str

def get_cs_dir(path_str, work_dir):
    # if not absolute, assume relative
    if not os.path.isabs(path_str):
        path_str = work_dir + '/' + path_str
    # Check if directory exists
    if not os.path.isdir(path_str):
        print('Error: directory does not exsist!')
        path_str = ""
    return path_str

def get_directories(arg_list):
    cs_dir = ""
    ts_dir = ""
    valid = True
    if len(arg_list) == 1:
        # No input directory provided
        print("Error: please provide a changeset directory")
    elif len(arg_list) == 2:
        # Must create a result directory...
        ts_dir = create_res_dir(work_dir)
        # Check if cs_dir exists
        cs_dir = arg_list[1]
        if not os.path.isdir(cs_dir):
            raise ValueError("Error: Changeset directory does not exist")
        else:
            valid, cs_dir = get_cs_dir(arg_list[1], work_dir)
    elif len(arg_list) == 3:
        # use result dir given
        ts_dir = create_res_dir(work_dir, arg_list[2])
        cs_dir = arg_list[1]
        if not os.path.isdir(cs_dir):
            raise ValueError("Error: Changeset directory does not exist")
        else:
            cs_dir = get_cs_dir(arg_list[1], work_dir)
    else:
        print("Error: too many arguments!")

    return cs_dir, ts_dir


if __name__ == '__main__':
    # COMMAND LINE ARGS
    #cs_dir = '/home/ubuntu/praxi/week5/cs_multitest'
    #ts_dir = '/home/ubuntu/praxi/week5/multitest_tags'

    work_dir = os.path.abspath('.')

    # Deal with command line arguments
    arg_list = sys.argv
    print(arg_list)
    cs_dir, ts_dir = get_directories(arg_list)

    if cs_dir == '':
        raise ValueError("Invalid changeset directory")
    if ts_dir == '':
        raise ValueError("Invalid tagset directory")

    # generate tagsets and place in ts directory!
    changeset_names = get_changeset_names(cs_dir)
    if len(changeset_names)==0:
        raise ValueError("No changesets in selected directory")

    tagset_names = create_tagset_names(changeset_names) # names for new tagset files!!!
    ids = get_ids(changeset_names)

    changesets = []
    labels = []
    changesets, labels = parse_cs(changeset_names, cs_dir, multilabel = True)

    tags = _get_columbus_tags(changesets)

    create_files(tagset_names, ts_dir, labels, ids, tags)
