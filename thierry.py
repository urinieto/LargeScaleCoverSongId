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
import pca      # Thierry's PCA
import dan_tools


PK_EXT=".pk"
PCA_EXT=".pca"
H5_EXT=".h5"
CHROMA_SUM_EXT="-ch-dict.txt.sum"
TONNETZ_SUM_EXT="-tz-dict.txt.sum"
INFTY = 10000000
PWR = 1.96
WIN = 75


def get_patches(feats, L=75):
    """Gets the 75-beat patches from feats."""
    N = feats.shape[0]
    patches = []

    if N < L:
        patch = np.zeros((L,feats.shape[1]))
        patch[:N,:] = feats
        patches.append(patch)
        return np.asarray(patches)

    for i in xrange(N):
        if i+L >= N: break
        patches.append(feats[i:i+L,:])
    return np.asarray(patches)

def get_MSD_path(track):
    """
    Gets the MSD path from a track.
        e.g. TRAAAZF12903CCCF6B --> A/A/A/TRAAAZF12903CCCF6B
    """
    return os.path.join(track[2], track[3], track[4], track)


def create_test_dataset(orig_path, dest_path, train_gt_file, N=500):
    """Creates a dataset of N songs extracted from the train set of the SHS."""
    train_gt = utils.read_pk(train_gt_file)
    cliques = train_gt.values()
    k = 0
    for clique in cliques:
        for track in clique:
            orig_file = os.path.join(orig_path, get_MSD_path(track) + PK_EXT)
            if not os.path.isfile(orig_file):
                orig_file = os.path.join(orig_path, "MSD", get_MSD_path(track) + PK_EXT)
            shutil.copy(orig_file, dest_path)
            k += 1
            if k == N: return


def create_test_dataset2(orig_path, dest_path, tracks_file, h5=False):
    """Creates a dataset from the tracks on the file."""
    tracks = []
    f = open(tracks_file, "r")
    for line in f.readlines():
        if line[0] == "%" or line[0] == "#":
            continue
        tracks.append(line.split("\n")[0])

    for track in tracks:
        if h5:
            print track
            orig_file = os.path.join(orig_path, get_MSD_path(track) + H5_EXT)
            data = extract_data(orig_file)
            new_file = os.path.join(dest_path, track + PK_EXT)
            pfile = open( new_file, "w" )
            cPickle.dump( data, pfile )
            pfile.close()
        else:
            orig_file = os.path.join(orig_path, get_MSD_path(track) + PK_EXT)
            if not os.path.isfile(orig_file):
                orig_file = os.path.join(orig_path, "MSD", get_MSD_path(track) + PK_EXT)
            shutil.copy(orig_file, dest_path)
    f.close()


def extract_data(filename):
    data = dict()
    data["pitches"] = dan_tools.msd_beatchroma(filename).T
    f = h5t.open_h5_file_read( filename )
    data["tonnetz"] = utils.chroma_to_tonnetz( data["pitches"] )
    data["track_id"] = h5t.get_track_id( f )
    data["artist_name"] = h5t.get_artist_name( f )
    data["title"] = h5t.get_title( f )
    f.close()
    return data


def compute_new_features(old_feats):
    """Computes the new features for the old_feats.

    Following Thierry's paper:
        1) Extract EN features
        2) Beat-sync
        3) Expenentiate by 1.96
        4) 2FTM each 75-beat patch
        5) Take Median
        6) PCA (not implemented in this function)
    """
    # 1) and 2)
    new_feats = old_feats
    # 3)
    #new_feats = np.power(new_feats, PWR)
    new_feats = dan_tools.chrompwr(new_feats.T, PWR)
    # 4)
    #patches = get_patches(new_feats, L=WIN)
    #patches_2dftm = np.asarray(utils.compute_fft2d_feats(patches))
    new_feats = dan_tools.btchroma_to_fftmat(new_feats, WIN)
    if new_feats is None:
        return None
    # 5)
    #median_2dftm = fftshift(np.median(patches_2dftm, axis=0))
    new_feats = np.median(new_feats, axis=1)
    print "new_feats", new_feats.shape
    median_2dftm = dan_tools.chromnorm(new_feats.reshape(new_feats.shape[0], 1))

    #plt.figure(1)
    #plt.imshow(np.reshape(median_2dftm, (75,12)).T, interpolation="nearest",
    #           aspect="auto")
    #plt.show()
    #print patches.shape, patches_2dftm.shape, median_2dftm.shape

    #return median_2dftm
    return median_2dftm.flatten()


def compute_new_sum_features(sum_feats):
    """Computes the new features for the old_feats.

    Following Thierry's paper:
        1) Extract EN features
        2) Beat-sync
        3) Expenentiate by 1.96
        4) 2FTM each 75-beat patch
        5) Take Median
        6) PCA (not implemented in this function)
    """
    for i,feat in enumerate(sum_feats):
        sum_feats[i] = dan_tools.chrompwr(feat.T, PWR).T

    sum_fft2d = utils.compute_fft2d_feats(sum_feats)
    for i,feat in enumerate(sum_fft2d):
        temp = fftshift(feat).flatten()
        caca = temp.reshape(temp.shape[0],1)
        sum_fft2d[i] = dan_tools.chromnorm(caca).flatten()

    #print len(sum_fft2d), np.asarray(sum_fft2d).shape, sum_fft2d[0].shape
    return sum_fft2d


def summarize_feats(feats, starts, N=16, key="pitches"):
    """Choose only the N beats from the starts of a set of features feats."""
    sum_feats = []
    for start in starts:
        sum_feats.append(feats[key][start:start+N,:])
        if len(sum_feats[-1]) != N:
            sum_feats[-1] = feats[key][-N:,:]
    return sum_feats


def compute_pca(X, PC=50):
    """Computes PCA from the X matrix of row observations using PC components."""
    X -= X.mean(axis=1)[:,np.newaxis] # Centered, per data point
    X -= X.mean(axis=0)[np.newaxis,:] # Centered, per feature coeff
    if PC != 0:
        print PC
        pca = PCA2(n_components=PC)
        try:
            return pca.fit_transform(X)
        except:
            print X.shape
    else: return X


def compute_new_features_from_list(files, PC=50, Tonnetz=False):
    """Compute the new features from the original Echo Nest features.
    files is the list of pickle files with the original beat-sync feats."""
    feats = []
    it = 0
    for file in files:
        old_feats = utils.read_pk(file)
        if Tonnetz:
            feats.append(compute_new_features(old_feats["tonnetz"]))
        else:
            feats.append(compute_new_features(old_feats["pitches"]))
        if it % 25 == 0: print "iteration:", it
        it += 1

    # Apply PCA
    #pca_feats = compute_pca(np.asarray(feats), PC=PC)

    # Uses thierry's PCA with trained data
    pcafile = "thierryCode/pca_250Kexamples_900dim_nocovers.pkl"
    trainedpca = utils.read_pk(pcafile)
    pca_feats = trainedpca.apply_newdata(feats, ndims=PC)

    # Save into files
    for file, feats in zip(files, pca_feats):
        f = open(file + PCA_EXT, "w")
        cPickle.dump(feats, f)
        f.close()


def parse_summary(summary):
    """Parses a music summary file."""
    f = open(summary, "r")
    lines = f.readlines()
    P = int(lines[1].split(":")[1].split("\n")[0])
    N = int(lines[2].split(":")[1].split("\n")[0])
    starts = map(int, lines[3].split("[")[1].split("]")[0].split(","))
    f.close()
    return P, N, starts


def compute_new_features_from_sum_list(files, PC=50, Tonnetz=False):
    """Compute the new features from the beat-synced Echo Nest features.
    files is the list of pickle files with the original beat-sync feats."""
    feats = []
    it = 0
    P = 0
    for i,file in enumerate(files):
        if Tonnetz:
            sum_file = file.replace(PK_EXT, TONNETZ_SUM_EXT)
        else:
            sum_file = file.replace(PK_EXT, CHROMA_SUM_EXT)
        old_feats = utils.read_pk(file)
        P, N, starts = parse_summary(sum_file)
        if Tonnetz:
            sum_feats = summarize_feats(old_feats, starts, N=N, key="tonnetz")
        else:
            sum_feats = summarize_feats(old_feats, starts, N=N, key="pitches")
        try:
            new_feats = compute_new_sum_features(sum_feats)
        except Exception, e:
            print i, file, e
            sys.exit()
            continue
        #print "e", new_feats
        for feat in new_feats:
            feats.append(np.asarray(feat))
        #print len(feats), feats[0]
        #if it % 25 == 0: print "iteration:", it
        it += 1

    # Apply PCA
    pca_feats = []
    if files != []:
        pca_feats = compute_pca(np.asarray(feats), PC=PC)
    else:
        print "Error: No files found!"

    final_feats = []
    feats = []
    for i,feat in enumerate(pca_feats):
        if i % P == 0:
            if feats != []:
                final_feats.append(feats)
            feats = []
        feats.append(feat)
    final_feats.append(feats)

    # Save into files
    for file, feats in zip(files, final_feats):
        f = open(file + PCA_EXT, "w")
        cPickle.dump(feats, f)
        f.close()


def feat_distance(f1, f2):
    """Compute the Euclidean distance between two feature vectors."""
    return np.linalg.norm(f1 - f2)


def dist_summaries1(pca1, pca2, pc=50):
    """Uses the Euclidean minimum distance between the subsequences of the summaries."""
    try:
        assert len(pca1) == len(pca2)
    except AssertionError:
        print "Assertion INFTY"
        return INFTY
    P = len(pca1)
    min_dist = INFTY
    for i in xrange(P):
        for j in xrange(P):
            #print pca1, pca2, len(pca1), len(pca2)
            d = np.linalg.norm(pca1[i][:pc] - pca2[j][:pc])
            if d < min_dist:
                min_dist = d
    return min_dist


def dist_summaries2(pca1, pca2, pc=50):
    """Uses the Euclidean average minimum distance between the subsequences of the summaries."""
    try:
        assert len(pca1) == len(pca2)
    except AssertionError:
        print "Assertion INFTY"
        return INFTY
    P = len(pca1)
    mins = []
    for i in xrange(P):
        min_dist = INFTY
        for j in xrange(P):
            #print pca1, pca2, len(pca1), len(pca2)
            d = np.linalg.norm(pca1[i][:pc] - pca2[j][:pc])
            if d < min_dist:
                min_dist = d
        mins.append(min_dist)
    return np.asarray(mins).mean()


def dist_summaries3(pca1, pca2, pc=50):
    """Uses the Euclidean average minimum distance between the subsequences of
    the summaries, and only one per tile."""
    try:
        assert len(pca1) == len(pca2)
    except AssertionError:
        print "Assertion INFTY"
        return INFTY
    P = len(pca1)
    mins = []
    D = np.zeros((P,P))
    for i in xrange(P):
        for j in xrange(P):
            D[i,j] = np.linalg.norm(pca1[i][:pc] - pca2[j][:pc])

    d = np.zeros(P)
    for p in xrange(P):
        d[p] = D.min()
        argmin = D.argmin()
        D[argmin/P,:] = INFTY
        D[:,argmin%P] = INFTY

    return d.mean()


def dist_summaries4(pca1, pca2):
    """Uses the Euclidean distance of the median of all the pcas."""
    try:
        assert len(pca1) == len(pca2)
    except AssertionError:
        print "Assertion INFTY"
        return INFTY
    P = len(pca1)
    #median1 = np.asarray( [coef for sublist in pca1 for coef in sublist] )
    #median2 = np.asarray( [coef for sublist in pca2 for coef in sublist] )
    median1 = np.median(np.asarray(pca1), axis=0)
    median2 = np.median(np.asarray(pca2), axis=0)
    return np.linalg.norm(median1 - median2)


def binary_evaluation_sum(path, tracks_file, distType=1, Tonnetz=False):
    """Binary evaluation of the summary files, following Thierry's paper."""

    f = open(tracks_file, "r")
    lines = f.readlines()
    it = 0
    correct = 0
    if distType == 1:
        dist_func = dist_summaries1
    elif distType == 2:
        dist_func = dist_summaries2
    elif distType == 3:
        dist_func = dist_summaries3
    elif distType == 4:
        dist_func = dist_summaries4

    if Tonnetz:
        ext = FileStrings.tonnetz_dict_base + ".sum" + PCA_EXT
    else:
        ext = FileStrings.chroma_dict_base + ".sum" + PCA_EXT

    for i,line in enumerate(lines):
        if line[0] == "%":
            try:
                featA = utils.read_pk(os.path.join(path,
                                                   lines[i+1].split("\n")[0] + ext))
                featB = utils.read_pk(os.path.join(path,
                                                   lines[i+2].split("\n")[0] + ext))
                featC = utils.read_pk(os.path.join(path,
                                                   lines[i+3].split("\n")[0] + ext))
            except IOError, e:
                print "IOError", line, e
                continue # Some summaries couldn't be computed
            dAB = dist_func(featA, featB)
            dAC = dist_func(featA, featC)
            if dAB < dAC:
                correct += 1
            it += 1
    f.close()
    res = correct / float(it)
    return res


def binary_evaluation(path, tracks_file):
    """Binary evaluation of the files, following Thierry's paper."""

    f = open(tracks_file, "r")
    lines = f.readlines()
    it = 0
    correct = 0
    for i,line in enumerate(lines):
        if line[0] == "%":
            featA = utils.read_pk(os.path.join(path, lines[i+1].split("\n")[0] + \
                                               PK_EXT + PCA_EXT))
            featB = utils.read_pk(os.path.join(path, lines[i+2].split("\n")[0] + \
                                               PK_EXT + PCA_EXT))
            featC = utils.read_pk(os.path.join(path, lines[i+3].split("\n")[0] + \
                                               PK_EXT + PCA_EXT))
            dAB = feat_distance(featA, featB)
            dAC = feat_distance(featA, featC)
            if dAB < dAC:
                correct += 1
            it += 1
    f.close()
    res = correct / float(it)
    print "Result of Binary Evaluation:", res


def process(dataset, PC=50, feats=False, sums=False, distType=1, Tonnetz=False):
    """Main process to reproduce Thierry's experiments."""
    # Compute the new features if needed
    if feats:
        print "Computing new features..."
        old_feat_files = glob.glob(os.path.join(dataset, "*" + PK_EXT))
        if sums:
            compute_new_features_from_sum_list(old_feat_files, PC, Tonnetz)
        else:
            compute_new_features_from_list(old_feat_files, PC, Tonnetz)

    # Binary Evaluation
    print "Evaluating..."
    new_feat_files = glob.glob(os.path.join(dataset,"*" + PCA_EXT))
    if sums:
        res = binary_evaluation_sum(dataset, "SHS/list_500queries.txt", distType)
    else:
        res = binary_evaluation(dataset, "SHS/list_500queries.txt")
    print "Result of Binary Evaluation:", res, PC, distType


def main():
    """Main function to find covers."""
    # Args parser
    parser = argparse.ArgumentParser(description=
             "Finds the covers given a summary within a summary data set.")
    parser.add_argument("dataset", action="store",
                        help="Path to the dataset")
    parser.add_argument("-p", action="store", dest="pc", default=50,
                        help="Number of Principal Components "\
                        "(default: 50)")
    parser.add_argument("-F", action="store_true", dest="feats",
                        help="Compute the features or not "\
                        "(default: False)")
    parser.add_argument("-S", action="store_true", dest="sums",
                        help="Use summaries, not original features "\
                        "(default: False)")
    parser.add_argument("-d", action="store", dest="distType", default=1,
                        choices=[1, 2, 3, 4],
                        help="Distance function type "\
                        "(default: 1)")
    parser.add_argument("-T", action="store_true", dest="tonnetz",
                        help="Use Tonnetz or not (default: False)")

    args = parser.parse_args()

    # Run the process
    start = time.time()
    process(dataset=args.dataset, PC=int(args.pc), feats=args.feats,
            sums=args.sums, distType=int(args.distType), Tonnetz=args.tonnetz)
    print "Time taken: %.3f sec"%(time.time() - start)


if __name__ == "__main__":
    main()
