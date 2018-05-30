import sys
import os
import optparse
import pickle
import logging
from .controller import discover_software_container
from .controller import index_files_from_list
from .controller import run_file_paths_discovery2
from .store.esclient import ESClient

IMG_WORKSPACE = "/Users/nagowda/Documents/columbus/imgworkspace"
DISCOVERY_OPTIONS = ['bin_names', 'file_paths', 'func-names',
                     'docker_history', 'pack_manager']


def main():
    logging.basicConfig(filename='./columbus.log', level=logging.INFO,
                        format='%(asctime)s %(message)s')

    usage = ("usage: python %prog -f <config_file> {--list | --discover } "
             "--route {bin_names | file_paths | func_names | docker_history "
             "| pack_manager}} <image_path>")
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-f", action="store", dest="systagfile",
                      help="System tag file")

    opts, args = parser.parse_args()
    if len(args) != 1:
        parser.error("incorrect number of arguments")

    systagfile = opts.systagfile
    imgpath = args[0]

    if not systagfile:
        parser.error("No config file specified")

    if not os.path.exists(IMG_WORKSPACE):
        os.makedirs(IMG_WORKSPACE)

    if not os.path.exists(imgpath):
        print("Invalid image\n")
        sys.exit(1)

    discover_software_container(imgpath, systagfile)
    # process = psutil.Process(os.getpid())
    # print "Memory Used %0.2f"%(float(process.get_memory_info().rss)/1000000)


def columbus(changeset, systagfile='/home/ubuntu/columbus/systags/ubuntu-1404'):
    """ Get labels from single changeset """
    esstore = ESClient('localhost', '9200')
    systags = {}
    with open(systagfile, 'rb') as sysfp:
        systags = pickle.load(sysfp)

    index_files_from_list(changeset, esstore)
    result = run_file_paths_discovery2("", systags['paths'], esstore)
    esstore.__del_index__("eureka")
    return result


if __name__ == "__main__":
    main()
