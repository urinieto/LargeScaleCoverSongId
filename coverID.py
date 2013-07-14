#!/usr/bin/env python

"""coverID.py: Identifies covers of a given summary within a summary data set."""

__author__      = "Uri Nieto"
__date__        = "25/11/12"

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
from pca import PCA
from pca import Center
from sklearn.decomposition import PCA as PCA2
from multiprocessing import Process, Queue
from sklearn.metrics import precision_recall_curve
import ranked

CORES = 8
PCA_MODEL = "pca.pk"
INFTY = 10000000


def pca_example2(feats, pc=50):
    """PCA using scikits learn."""
    A = np.zeros((len(feats), feats[0].shape[0]*feats[0].shape[1]))
    for i,feat in enumerate(feats):
        A[i,:] = feat.flatten()
    Center(A)
    pca = PCA2(n_components=pc)
    pca.fit(A)
    print "PC:", pca.explained_variance_ratio_, A.shape, pca.components_, pca.transform(A).shape
    return A


def pca_example(A, pc=8):
    """PCA using Stack Overflow example:
    http://stackoverflow.com/questions/1730600/principal-component-analysis-in-python
    """
    print "PCA ..." ,
    Center(A)   # Scale data (mean and std)
    p = PCA( A, fraction=0.99 )
    print "npc:", p.npc
    print "% variance:", p.sumvariance * 100

    print "Vt[0], weights that give PC 0:", p.Vt[0]
    print "A . Vt[0]:", np.dot( A, p.Vt[0] )
    print "pc:", p.pc()

    print "\nobs <-> pc <-> x: with fraction=1, diffs should be ~ 0"
    x = np.ones(A.shape[1])
    # x = np.ones(( 3, K ))
    print "x:", x
    pc = p.vars_pc(x)  # d' Vt' x
    print "vars_pc(x):", pc
    print "back to ~ x:", p.pc_vars(pc)

    Ax = np.dot( A, x.T )
    pcx = p.obs(x)  # U' d' Vt' x
    print "Ax:", Ax
    print "A'x:", pcx
    print "max |Ax - A'x|: %.2g" % np.linalg.norm( Ax - pcx, np.inf )

    b = Ax  # ~ back to original x, Ainv A x
    back = p.vars(b)
    print "~ back again:", back
    print "max |back - x|: %.2g" % np.linalg.norm( back - x, np.inf )
    print "eigen", p.eigen


def compute_pca_feats(feats, npc=50, model_pk=PCA_MODEL):
    """Computes the npc PCA components of the features feats, using the model
    model_pk."""
    #pca_example(feats[0])
    #A = pca_example2(feats)
    #pca_example(A)

    # Read model
    p = utils.read_pk(model_pk)
    # set number of principal components
    p.npc = npc

    # Compute for each subsequence
    pca_feats = []
    for feat in feats:
        # transform into PC space
        pca_feats.append(p.vars_pc(feat))

    return pca_feats


def compute_final_feats(sum_feats, pc=50):
    """Computes the 2DFFT and PCA on the sum_feats."""
    fft_feats = utils.compute_fft2d_feats(sum_feats)
    #plt.figure(1)
    #num = 1
    #fft_feats[num][0,0] = 0
    #plt.imshow(sum_feats[num], interpolation="nearest", aspect="auto")
    #plt.figure(2)
    #plt.imshow(fft_feats[num], interpolation="nearest", aspect="auto")
    #plt.show()
    #print fft_feats[0].shape, sum_feats[0].shape, fft_feats
    pca_feats = compute_pca_feats(fft_feats, pc)
    return pca_feats


def summarize_feats(feats, starts, N=16, key="pitches"):
    """Choose only the N beats from the starts of a set of features feats."""
    sum_feats = []
    for start in starts:
        sum_feats.append(feats[key][start:start+N,:])
    return sum_feats


def read_pk(pickle_file):
    """Reads the pickle file and returns its content."""
    f = open(pickle_file, "r")
    data = cPickle.load(f)
    f.close()
    return data


def get_sum_feats(summary, dataset, starts, N=16):
    """Gets the summarized features from a pickle file that has the same TRACK ID than
    the summary, that can be found in the dataset path."""
    track = os.path.basename(summary)
    track = track.split("-")[0] + ".pk"
    pk_file = os.path.join(dataset, track[2], track[3],
                                track[4], track)
    # Hack for when the distances are still being computed
    try:
        with open(pk_file) as f: pass
    except IOError as e:
        print 'Oh dear.'
        root_dir = pk_file.split("/")[0]
        pk_file = pk_file[:len(root_dir)+1] + "MSD/" + pk_file[len(root_dir)+1:]
    all_feats = utils.read_pk(pk_file)
    # TODO: Tonnetz also?
    sum_feats = summarize_feats(all_feats, starts, N, key="pitches")
    return sum_feats


def parse_summary(summary):
    """Parses a music summary file."""
    f = open(summary, "r")
    lines = f.readlines()
    P = int(lines[1].split(":")[1].split("\n")[0])
    N = int(lines[2].split(":")[1].split("\n")[0])
    starts = map(int, lines[3].split("[")[1].split("]")[0].split(","))
    f.close()
    return P, N, starts


def train_PCA(summaries, dataset, ntrain=1000, out=PCA_MODEL):
    """Trains a PCA with ntrain observations, and saves the PCA class into
    the output file."""
    assert ntrain <= 10000
    # TODO: Train with Tonnetz!
    sums = glob.glob(os.path.join(summaries, "*-ch-*"))
    # TODO: Don't hardcode!
    P = 4
    N = 16
    # Construct input matrix
    X = np.zeros((ntrain*P,N*12))
    for i in range(ntrain):
        P, N, starts = parse_summary(sums[i])
        sum_feats = get_sum_feats(sums[i], dataset, starts, N)
        try:
            fft_feats = compute_fft2d_feats(sum_feats)
        except ValueError:
            print "Error: No data in", sums[i]
            continue
        for p in range(P):
            try:
                X[i+p,:] = fft_feats[p]
            except ValueError:
                print "Error: Summary reaches the end", sums[i]
    # Remove errors
    errors = np.where(np.all(X==0))[0]
    X = np.delete(X, errors, axis=0)

    # PCA
    Center(X)   # Scale data (mean and std)
    p = PCA( X, fraction=0.99 )

    # Save model
    f = open(out, "w")
    cPickle.dump(p,f)
    f.close()


def process_summaries(q, sum_list, dataset, pc):
    """Processes a list of summaries, helpful when multi-coreing."""
    for s in sum_list:
        P, N, starts = parse_summary(s)
        sum_feats = get_sum_feats(s, dataset, starts, N)
        try:
            final_feats = compute_final_feats(sum_feats, pc=pc) # 2DFFT + PCA
        except Exception, e:
            print "Error: couldn't compute the new features for", s
            final_feats = []

        #save file:
        print s+".pca"
        f = open(s + ".pca", "w")
        cPickle.dump(final_feats,f)
        f.close()


def dist_summaries(pca1, pca2, pc=50):
    """Uses the Euclidean minimum distance between the subsequences of the summaries."""
    try:
        assert len(pca1) == len(pca2)
    except AssertionError:
        return INFTY
    P = len(pca1)
    min_dist = INFTY
    for i in xrange(P):
        for j in xrange(P):
            d = np.linalg.norm(pca1[i][:pc] - pca2[j][:pc])
            if d < min_dist:
                min_dist = d
    return min_dist


def clean_list(gt, l):
    """Cleans the list from the files that are not found on the GT."""
    new_l = []
    for f in l:
        if get_key_for_track(gt,os.path.basename(f).split("-")[0]) != 0:
            new_l.append(f)
    return new_l


def process_pca(q, pca_list, dataset, pc, shs_gt):
    """Processes a list of PCA coefficients and computes the Average
    Precision."""
    clique_gt = utils.read_pk(shs_gt)
    track_gt = utils.compute_track_from_clique_dict(clique_gt)
    #pca_list = clean_list(gt, pca_list) # Already cleaned!
    NumPCA = len(pca_list)
    MAP = 0
    k = 0
    for i in xrange(NumPCA):
        # Compute the distances
        S = np.zeros(NumPCA)    # Array of Scores
        L = np.zeros(NumPCA)    # Array of keys from GT
        query = utils.read_pk(pca_list[i])
        for j in xrange(NumPCA):
            L[j] = get_key_for_track(
                    track_gt, os.path.basename(pca_list[j]).split("-")[0])
            if i==j:
                S[i] = INFTY
                continue
            x = utils.read_pk(pca_list[j])
            S[j] = dist_summaries(query, x, pc=pc)
        # Compute the Average Precision
        cliqueID = get_key_for_track(
                    track_gt, os.path.basename(pca_list[i]).split("-")[0])
        I = np.argsort(S)
        cliqueID_i = int(cliqueID)
        estimated = (cliqueID_i == L[I])[:-1] # Don't take the last one (it's the query)
        print estimated.shape
        recall, precision = ranked.ordered_recall_precision(estimated)
        int_recall, int_precision = \
            ranked.maxinterp_recall_precision(recall, precision,
                                              n_points=15, DISC=False)

        k += 1

        MAP += int_precision.mean()
        print "precision:", precision, "recall:", recall
        print "interp precision:", int_precision, "interp recall:", int_recall
        print "Mean:", int_precision.mean()
        print "MAP:", MAP / float(k)


def get_key_for_track(gt, track):
    """Gets the clique ID for a given track for a SHS GT."""
    #for key in gt.keys():
    #    for t in gt[key]:
    #        if t == track:
    #            return key
    #print "Error: Key not found in GT!"
    try:
        return gt[track]
    except:
        print "Not found"
    return 0


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def process(sum_path, dataset, shs_gt="SHS/train.pk", shs=False, pc=8, ntrain=0):
    """Main process to find covers within a data set."""
    if ntrain != 0:
        train_PCA(sum_path, dataset, ntrain)
        return
    if shs:
        # TODO: Get Tonnetz too?
        sum_list = glob.glob(os.path.join(sum_path, "*-ch-dict.txt.sum"))

        # Multi-processing
        chunk_sums = list(chunks(sum_list, len(sum_list)/(CORES-1)))
        q = Queue()
        P = []
        # Create Processes
        for c in range(CORES):
            P.append(Process(target=process_summaries,
                             args=(q,chunk_sums[c], dataset, pc)))
        # Run Processes
        for c in range(CORES):
            P[c].start()

    q = Queue()
    pca_list = glob.glob(os.path.join(sum_path, "*-ch-dict.txt.sum.pca"))
    chunk_pcas = list(chunks(pca_list, len(pca_list)))
    process_pca(q, chunk_pcas[0], dataset, pc, shs_gt)


def main():
    """Main function to find covers."""
    # Args parser
    parser = argparse.ArgumentParser(description=
             "Finds the covers given a summary within a summary data set.")
    parser.add_argument("sum_path", action="store",
                        help="Path to the summaries")
    parser.add_argument("dataset", action="store",
                        help="Path to the dataset")
    parser.add_argument("shs_gt", action="store",
                        help="Pickle file of the SHS ground truth")
    parser.add_argument("-s", action="store_true", dest="shs",
                        help="The dataset is a SHS dataset "\
                        "(default: False)")
    parser.add_argument("-p", action="store", dest="pc", default=100,
                        help="Number of Principal Components "\
                        "(default: 100)")
    parser.add_argument("-T", action="store", dest="ntrain", default=0,
                        help="Train the PCA with NTRAIN numbers"\
                        "(default: No training)")
    args = parser.parse_args()

    # Run the process
    start = time.time()
    process(sum_path=args.sum_path, dataset=args.dataset, shs_gt=args.shs_gt,
            shs=args.shs, pc=int(args.pc), ntrain=int(args.ntrain))
    print "Time taken: %.3f sec"%(time.time() - start)


if __name__ == "__main__":
    main()
