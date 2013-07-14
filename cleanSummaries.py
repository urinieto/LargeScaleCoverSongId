#!/usr/bin/env python

"""cleanSummaries.py: Moves to the correct place the SHS summaries."""

__author__      = "Uri Nieto"
__date__        = "26/11/12"

import time
import argparse
import os
import glob
import numpy as np
import utils
import cPickle
import subprocess
from __init__ import FileStrings
from scipy.spatial import distance
from pca import PCA
from pca import Center
from sklearn.decomposition import PCA as PCA2
from multiprocessing import Process, Queue
import shutil


def read_pk(pickle_file):
    """Reads the pickle file and returns its content."""
    f = open(pickle_file, "r")
    data = cPickle.load(f)
    f.close()
    return data


def exists_track(gt, track):
    """Whether the track exists in the GT or not."""
    for key in gt.keys():
        for t in gt[key]:
            if t == track:
                return True
    return False


def move_files(files_list, src_folder, dst_folder, gt):
    """Moves the files found in GT from orig folder into dest folder."""
    for f in files_list:
        base =  os.path.basename(f).split("-")[0]
        if exists_track(gt, base):
            to_move = glob.glob(os.path.join(src_folder,base + "*"))
            for src in to_move:
                shutil.move(src, dst_folder)


def process(sum_path):
    train_gt = read_pk(os.path.join(sum_path, "train.pk"))
    test_gt = read_pk(os.path.join(sum_path, "test.pk"))
    train_folder = os.path.join(sum_path, "train-summaries")
    test_folder = os.path.join(sum_path, "test-summaries")
    bad_folder = os.path.join(sum_path, "bad-summaries")
    files_list = glob.glob(os.path.join(bad_folder, "*-ch-dict.txt.sum.pca"))
    move_files(files_list, bad_folder, train_folder, train_gt)
    move_files(files_list, bad_folder, test_folder, test_gt)


def main():
    """Main function to find covers."""
    # Args parser
    parser = argparse.ArgumentParser(description=
             "Finds the covers given a summary within a summary data set.")
    parser.add_argument("sum_path", action="store",
                        help="Path to the summaries")
    args = parser.parse_args()

    # Run the process
    start = time.time()
    process(sum_path=args.sum_path)
    print "Time taken: %.3f sec"%(time.time() - start)


if __name__ == "__main__":
    main()
