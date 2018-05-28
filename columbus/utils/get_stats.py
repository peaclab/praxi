import os
import sys
import optparse
from store.esclient import ESClient
import pdb

def main():
    usage = "usage: python %prog -f <config.cfg> --layerid <layerid> --tag <tag>"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-l", "--layerid",  action="store", dest="layerid", help="layerid to inspect")
    parser.add_option("-t", "--tag",  action="store", dest="tag", help="tag to stat for")
    parser.add_option("-f", action="store", dest="cfgfile", help="Config file")
    opts,args= parser.parse_args()

    layerid = opts.layerid
    tag = opts.tag
    esstore = ESClient('localhost', '9200')
    (total_exec, matched_exec) = esstore.get_executables_stats('eureka', layerid, tag) 
    print "Total executables = %d Executables taged for software [%s] = %d"%(total_exec, tag,  matched_exec)
    (total_fpaths, matched_fpaths) = esstore.get_filepaths_stats('eureka', layerid, tag) 
    print "Total files = %d filepaths taged for software [%s] = %d"%(total_fpaths, tag, matched_fpaths)

if __name__=="__main__":
    main()
