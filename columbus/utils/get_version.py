import os
import sys
import argparse
import commands
import shutil
import pdb

REPO_PATH = "/home/shripad/imgrepo/"
COLUMBUS_HOME = "/home/shripad/columbus/columbus"
COLUMBUS_WORKSPACE = "/home/imgworkspace/*"

def exec_cmd(cmd):
    status, output = commands.getstatusoutput(cmd)
    if status !=0 :
        print "Error executing command. %s"%(cmd)
        print output
    return status, output.split('\n')

def main():
    parser =  argparse.ArgumentParser(description='Software version analysis')
    parser.add_argument('-i', '--input', action = "store", dest="infile")
    parser.add_argument('-o', '--outdir', action = "store", dest="outdir")

    opts = parser.parse_args()
    infile = opts.infile
    outdir = opts.outdir

    if not os.path.exists(infile):
        print "File not found. %s"%(infile)
        sys.exit(1)

    fin = open(infile, "r")
    for line in fin:
        if line.startswith('#'):
            continue

        softs = line.strip().split(',')
        imgname = softs[0]
        if len(softs) > 1:
            tags = softs[1:]
        else:
            continue
   
        #0. Clear ES Index
        cmd = "curl -XDELETE http://localhost:9200/_all"
        rc,_ = exec_cmd(cmd)
        if rc !=0 :
            print "Unable to clear ES Index. Exiting..."
            sys.exit(1)

        #1. save the image into tag.gz
        cmd = "docker save {IMAGE} > {IMGREPO}/{IMAGE}.tar.gz".format(IMAGE =
                                                                   imgname,\
                                                                   IMGREPO =
                                                                   REPO_PATH)
        print cmd
	rc, _ = exec_cmd(cmd)
        if rc != 0:
            print "Error saving image %s. Skipping this image."%(imgname)
            continue

        #2. Run Columbus and get discovery result
        imgpath = "{IMGREPO}/{IMAGE}.tar.gz".format(IMAGE = imgname, IMGREPO =
                                                  REPO_PATH)
        cmd = "python {CHOME}/columbus.py -f {CHOME}/systags/ubuntu-1404 \
        --discover --route file_paths {IMGPATH}".format(CHOME = COLUMBUS_HOME, IMGPATH = imgpath)

        rc, out = exec_cmd(cmd)
        if rc != 0:
            print "Columbus failed to discover tags for image %s. Skipping this \
            image"(imgpath)
            continue 

        #3. Read each line of discovery result and search for discovered tags
        #   and find layer-id
        for outline in out:
            tagfound = False
            for tag in tags:
                if outline.find(tag) < 0 : #tag not found in output
                    continue
                else:
                    tagfound = True
                    break
            if tagfound:
                resdir = "{OUT_REPO}/{SOFTNAME}".format(OUT_REPO = outdir,
                                                      SOFTNAME = imgname)
                if not os.path.exists(resdir):
                    os.makedirs(resdir)

                layerid = outline.split('\t')[0]
                ofile = os.path.join(resdir, layerid)

                #3.1 for each layer-id, run print_all_paths to get list of filepaths 
                cmd = "python {CHOME}/print_all_paths.py -l {LAYERID} -o {RES_FILE} -m \"{MSG}\"".format(CHOME = COLUMBUS_HOME, LAYERID = layerid,
                                 RES_FILE = ofile, MSG = outline)

                rc, _ = exec_cmd(cmd)
                if rc != 0:
                    print "Unable to print file paths for image = %s layer =%s"%(imgname, layerid)
                    continue

        #4. Cleanup ; Remove img tar file, remove image from columbus workspace
        cmd = "rm -rf %s"%(imgpath)
        rc, _ = exec_cmd(cmd)

        cmd = "rm -rf %s"%(COLUMBUS_WORKSPACE)
        rc, _ = exec_cmd(cmd)


    fin.close()    

if __name__=="__main__":
    main()

