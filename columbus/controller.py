import pickle
from .pytrie.trie import Trie
from datetime import datetime
import glob
import yaml

IMG_WORKSPACE = "/Users/nagowda/Documents/columbus/imgworkspace"
FILTER_PATH = ['.wh..wh.plnk', '.wh..wh.orph', 'layer.tar', '.wh..wh.auf']
ES_BATCH_SIZE = 1000
FILTER_PATH_TOKENS = ['usr', 'bin', 'proc', 'sys', 'etc', 'local', 'src',
                      'dev', 'home', 'root', 'lib', 'pkg', 'sbin', 'share',
                      'cache']


def run_file_paths_discovery2(filtertags, cfilePath):
    layerid = "1"

    ftrie = Trie()
    with open(cfilePath, "r") as cStream:
        changeset = yaml.load(cStream)

    chfiles = changeset['changes']
    pkgLabel = changeset['label']

    for filepath in chfiles
        pathtokens = filepath.split('/')
        for token in pathtokens:
            if token != '' and token not in FILTER_PATH_TOKENS:
                ftrie.insert(token)

    softtags = {}
    res = ftrie.get_all_tags()
    for tag in res:
        if tag in filtertags:
            continue
        softtags[tag] = res[tag]
    return softtags
