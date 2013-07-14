#!/usr/bin/env python

"""thierry.py: Replicates (or at least tries to replicate) Thierry's paper."""

__author__      = "Uri Nieto"
__date__        = "27/11/12"

import time
import argparse
import os
import glob
import pylab as plt
import numpy as np
import utils
import cPickle
import subprocess
from __init__ import FileStrings
from scipy.spatial import distance
#from pca import PCA
#from pca import Center
from sklearn.decomposition import PCA as PCA2
from multiprocessing import Process, Queue
from sklearn.metrics import precision_recall_curve
import ranked
import shutil
import sys
from scipy.fftpack import fftshift
import hdf5_getters as h5t
import thierry as T

def process(dataset, Tonnetz=False):
    PC = [0, 100, 50, 20, 10, 5]
    D = [1,2,3,4]
    #D = [3]
    for pc in PC:
        for d in D:
            T.process(dataset, PC=pc, feats=True, sums=True, distType=d,
                      Tonnetz=Tonnetz)

def main():
    """Main function to find covers."""
    # Args parser
    parser = argparse.ArgumentParser(description=
             "Finds the covers given a summary within a summary data set.")
    parser.add_argument("dataset", action="store",
                        help="Path to the dataset")
    parser.add_argument("-T", action="store_true", dest="tonnetz",
                        help="Use Tonnetz or not (default: False)")

    args = parser.parse_args()

    # Run the process
    start = time.time()
    process(dataset=args.dataset, Tonnetz=args.tonnetz)
    print "Time taken: %.3f sec"%(time.time() - start)


if __name__ == "__main__":
    main()

