import os
import hashlib
import json
import shutil
import tarfile
import pdb
import pickle
import sys
import logging
import tempfile
import stat
from store.esclient import ESClient
from pytrie.trie import Trie
import re
from datetime import datetime
import glob
import yaml

IMG_WORKSPACE = "/Users/nagowda/Documents/columbus/imgworkspace"
FILTER_PATH = ['.wh..wh.plnk','.wh..wh.orph', 'layer.tar', '.wh..wh.auf']
ES_BATCH_SIZE = 1000
FILTER_PATH_TOKENS = ['usr', 'bin', 'proc', 'sys', 'etc', 'local', 'src','dev','home', 'root', 'lib', 'pkg', 'sbin', 'share','cache']

def index_files(fname, esstore):
    with open(fname, "r") as cStream:
        changeset = yaml.load(cStream)

    chfiles = changeset['changes']
    try:
       # modifiedfile = changeset['modifications']
       pkgLabel = changeset['label']
    except KeyError:
       pkgLabel = "Not found: %s"%(fname)

    index_file_from_list(chfiles, esstore)
    return pkgLabel


def index_file_from_list(chfiles, esstore):
    layerid = "1"
    flist = []
    for cfile in chfiles:
        # pdb.set_trace()
        filemd = cfile.split()
        # if filemd[0] == '000':
        #     continue
        # else:
        #   fsmd = {}
        fsmd = {}
        fsmd['path'] = filemd[1]
        fsmd['_type'] = 'file'
        fsmd['_index'] = "eureka"
        fsmd['layer'] = layerid
        flist.append(fsmd)

    esstore.__insert_fs_metdata__("eureka", layerid, flist)


def discover_software_container(changesetDirpath, systagfile):
    esstore = ESClient('localhost', '9200')
    systags = {}
    with open(systagfile, 'rb') as sysfp:
        systags = pickle.load(sysfp)

    progress = 1
    testrepo = glob.glob(changesetDirpath+"/*")
    skip = 0
    for fname in testrepo:
        t1 = datetime.now()
        print("[INFO]Filepath = %s" % (fname))
        print("[INFO]Processing package %d/%d" % (progress, len(testrepo)))
        progress += 1
        if skip > 0:
            skip -= 1
            continue
        # pkgName = index_files(os.path.join(changesetDirpath, fname), esstore)
        pkgName = index_files(fname, esstore)
        run_file_paths_discovery2(pkgName, systags['paths'], esstore)
        esstore.__del_index__("eureka")
        t2 = datetime.now()
        print("Time taken: ", (t2-t1))


def run_file_paths_discovery2(pkgName, filtertags, esstore):
    layerid = "1"

    files = esstore.__get_all_files__("eureka", layerid)
    ftrie = Trie()
    for filepath in files:
        pathtokens = filepath.split('/')
        for token in pathtokens:
            if token != '' and token not in FILTER_PATH_TOKENS:
                ftrie.insert(token)

    softtags = []
    k = 15 * 5  # Emre: changed this to output more tags for multi-app runs
    res = ftrie.get_all_tags()
    for tag in res:
        if tag in filtertags:
            continue
        # if res[tag] < 2:
        #     break
        softtags.append(tag)
        k -= 1
        if k == 0:
            break

    if pkgName != "":
        saveToFile(pkgName, softtags)
        print("%s\t%s" % (pkgName, softtags))
    return softtags


def saveToFile(pkgName, softtags):
    fp = open("result", "a+")
    res="%s\t%s\n"%(pkgName, softtags)
    fp.write(res)
    fp.close()

