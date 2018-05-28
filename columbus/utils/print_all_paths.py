import os
import sys
import optparse
from store.esclient import ESClient
import pdb
from pytrie.trie import Trie

FILTER_PATH_TOKENS = ['usr', 'bin', 'proc', 'sys', 'etc', 'local', 'src','dev','home', 'root', 'lib', 'pkg', 'sbin', 'share','cache', 'yumdb', 'yum']

def main():
    usage = "usage: python %prog -f <config.cfg> --layerid <layerid> --tag <tag>"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-l", "--layerid",  action="store", dest="layerid", help="layerid to inspect")
    parser.add_option("-m", "--msg",  action="store", dest="msg", help="message")
    parser.add_option("-o", "--output",  action="store", dest="outfile",
                      help="output file")
    opts,args= parser.parse_args()

    layerid = opts.layerid
    outfile = opts.outfile
    msg = opts.msg

    fout = open(outfile, 'w+')
    fout.write(msg)
    fout.write('\n')

    esstore = ESClient('localhost', '9200')
    files =  esstore.__get_all_files__('eureka', layerid)
    for filepath in files:
        fout.write(filepath)
        fout.write("\n")

    fout.close()

if __name__=="__main__":
    main()
