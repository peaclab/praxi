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
    return status, output

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
        rc, _ = exec_cmd(cmd)
        if rc != 0:
            print "Error saving image %s. Skipping this image."%(imgname)
            continue

        #2. Run Columbus and get discovery result
        imgpath = "{IMGREPO}/{IMAGE}.tar.gz".format(IMAGE = imgname, IMGREPO =
                                                  REPO_PATH)
        cmd = "python {CHOME}/columbus.py -f {CHOME}/systags/ubuntu-1404 \
        --discover --route bin_names {IMGPATH}".format(CHOME = COLUMBUS_HOME, IMGPATH = imgpath)

        rc, out = exec_cmd(cmd)
        if rc != 0:
            print "Columbus failed to discover tags for image %s. Skipping this \
            image"(imgpath)
            continue 

        #3. Read each line of discovery result and search for discovered tags
        #   and find layer-id
        if True:
                ofile = "{OUT_REPO}/{SOFTNAME}".format(OUT_REPO = outdir,
                                                      SOFTNAME = imgname)
                with open(ofile, 'w+') as fout:
                    fout.write(out)

        #4. Cleanup ; Remove img tar file, remove image from columbus workspace
        cmd = "rm -rf %s"%(imgpath)
        rc, _ = exec_cmd(cmd)

        cmd = "rm -rf %s"%(COLUMBUS_WORKSPACE)
        rc, _ = exec_cmd(cmd)


    fin.close()    

if __name__=="__main__":
    main()

