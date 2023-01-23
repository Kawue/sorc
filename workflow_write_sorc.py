import sys
sys.path.append('.')
import argparse
import os
from sorc import write_sorc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--h5path", type=str, required=True, help="Path to SORC stats HDF5 file.")
parser.add_argument("-s", "--savepath", type=str, required=False, default=None, help="Path to save HTML SORC output (with .html extension).")
parser.add_argument("-l", "--limit", type=int, required=False, default=None, help="Limits the number of displayed values.")
args=parser.parse_args()

h5pypath = args.h5path
if type(args.savepath) == type(None):
    savepath = os.path.join(os.path.dirname(h5pypath), os.path.basename(h5pypath).split(".h5")[0] + ".html")
else:
    savepath = args.savepath
limit = args.limit

if not os.path.exists(os.path.dirname(savepath)):
    os.makedirs(os.path.dirname(savepath))

write_sorc(h5pypath, savepath, limit)