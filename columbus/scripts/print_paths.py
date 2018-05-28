import os
import sys
import optparse
from store.esclient import ESClient
import pdb
import re

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    usage = "usage: python %prog -f <config.cfg> --layerid <layerid> --tag <tag>"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-l", "--layerid",  action="store", dest="layerid", help="layerid to inspect")
    parser.add_option("-t", "--tag",  action="store", dest="tag", help="tag to stat for")
    parser.add_option("-o", action="store", dest="op", help="Search option")
    opts,args= parser.parse_args()

    layerid = opts.layerid
    tag = opts.tag
    op = opts.op

    esstore = ESClient('localhost', '9200')
 
    if op == 'file':
        pathlist = esstore.get_all_matched_files('eureka', layerid, tag)
        for path in pathlist:
            print path.replace(tag, bcolors.OKGREEN+tag+bcolors.ENDC)
    elif op == 'bin':
        binlist = esstore.get_all_matched_execs('eureka', layerid, tag)
        for bin in binlist:
            print bin.replace(tag, bcolors.OKGREEN+tag+bcolors.ENDC)

if __name__=="__main__":
    main()
