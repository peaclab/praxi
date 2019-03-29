import argparse
import os

parser = argparse.ArgumentParser(description='Arguments for Praxi software discovery algorithm.')
parser.add_argument('-tr','--traindir', help='Path to training tagset directory.', required=True)
parser.add_argument('-ts', '--testdir', help='Path to testing tagset directoy.', required=True)
parser.add_argument('-od', '--outdir', help='Path to desired result directory', default='.')
# run a single label experiment by default, if --multi flag is added, run a multilabel experiment!
parser.add_argument('--multi', dest='experiment', action='store_const', const='multi', default='single', help="Type of experiment to run (single-label default).")
parser.add_argument('-vw','--vwargs', dest='vw_args', default='-b 26 --learning_rate 1.5 --passes 10', help="custom arguments for VW.")

#'-b 26 --learning_rate 1.5 --passes 10'
# default single, arg to do multi instead

#parser.add_argument('--traindir', dest='accumulate', action='store_const',
#               const=sum, default=max,
#               help='sum the integers (default: find the max)')

args = vars(parser.parse_args())
print(args)

exp_type = args['experiment'] # 'single' or 'multi'
print("exp type = ", exp_type)

ts_train_path = args['traindir']

outdir = os.path.abspath(args['outdir'])
print(type(outdir))
print("outdir", outdir)
