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
    parser.add_option("-t", "--tag",  action="store", dest="tag", help="tag to stat for")
    parser.add_option("-o", action="store", dest="mode", help="mode (file/bin)")
    opts,args= parser.parse_args()

    layerid = opts.layerid
    tag = opts.tag
    mode = opts.mode

    esstore = ESClient('9.12.235.150', '9200')
    ftrie = Trie()
    tokens = 0
    if mode == 'file': 
        files =  esstore.__get_all_files__('eureka', layerid)
        for filepath in files:
            pathtokens = filepath.split('/')
            for token in pathtokens:
                if token != '' and not token in FILTER_PATH_TOKENS:
	            tokens+=1		
                    ftrie.insert(token)

    else:
        binfiles =  esstore.__get_all_binaries__('eureka', layerid)       
        if len(binfiles) <= 3:
	    for binary in binfiles:
		bname = os.path.basename(binary)
		tokens+=1
		ftrie.direct_tag_insert(bname)

        else:	 
    	    for binary in binfiles:
		tokens+=1
		bname = os.path.basename(binary)
		ftrie.insert(bname)

    res = ftrie.get_all_tags()
    rank = 1
    for ftag in res:
        print "%s %d"%(ftag, res[ftag])

if __name__=="__main__":
    main()
