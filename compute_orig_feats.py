#!/usr/bin/env python
"""
Computes various features from the SHS dataset.

Uri Nieto, April 25th 2013
"""


import os
import sys
import cPickle
import numpy as np
import argparse
import time
import random
# local stuff
import utils
import dan_tools


def compute_original_feats(maindir, tracks, cliques, output="originalfeats.pk"):
    """Computes the original features."""
    X = []
    I = []
    cnt = 0
    k = 0
    for tid in tracks:
        path = utils.path_from_tid(maindir, tid)
        feats = utils.extract_feats(path)
        if feats == None:
            continue
        x = np.zeros(feats.shape)
        for i,feat in enumerate(feats):
            #feat = dan_tools.chromnorm(feat.reshape(feat.shape[0], 1)).squeeze()
            #feat = np.reshape(feat, (1,900))
            x[i] = feat
        X.append(x)

        for i, clique in enumerate(cliques):
            if tid in clique:
                idx = i
                break
        I.append(idx)

        if cnt % 50 == 0 and cnt != 0:
            print "---Computing features: %d of %d" % (cnt, len(tracks))
            f = open("/Volumes/Audio/SummaryCovers/originalfeats%d.pk"%k, 'w')
            cPickle.dump((X,I), f)
            f.close()
            k += 1
            X = []
        cnt += 1

def compute_clique_idxs(cliques, tracks, output="clique_idx.pk"):
    """Computes the clique indeces."""
    clique_idx = []
    for t, tid in enumerate(tracks):
        if t == 244: print tid
        idx = -1
        for i, clique in enumerate(cliques):
            if tid in clique:
                idx = i
                break
        clique_idx.append(idx)
        assert idx != -1, "ERROR computing clique idxs"
    f = open(output, 'w')
    cPickle.dump(np.asarray(clique_idx), f)
    f.close() 


def compute_one_clique(maindir, cliques, mu, sd, clique_id=0,
                       output="clique_vs_nonclique.pk"):
    """Computes the features for one clique, and N other tracks as 
        non_cliques."""
    X = dict()
    X["cliques"] = []
    X["non_cliques"] = []
    for tid in cliques[clique_id]:
        path = utils.path_from_tid(maindir, tid)
        feats = utils.extract_feats(path)
        if feats == None:
            continue
        x = np.zeros((feats.shape[0], feats.shape[1]/2))
        for i,feat in enumerate(feats):
            x[i] = feat[450:]
        X["cliques"].append(x)

    N = len(cliques[clique_id])
    n = 0
    checked_cliques = []
    checked_cliques.append(clique_id)
    while n < N:
        idx = np.random.random_integers(0,len(cliques))
        if idx in checked_cliques:
            continue
        path = utils.path_from_tid(maindir, cliques[idx][0])
        feats = utils.extract_feats(path)
        if feats == None:
            continue
        x = np.zeros((feats.shape[0], feats.shape[1]/2))
        for i,feat in enumerate(feats):
            x[i] = feat[450:]
        n += 1
        X["non_cliques"].append(x)

    feats = np.empty((0,450))
    bounds = []
    for key in X:
        print key
        for x in X[key]:
            x = standardize(x, mu, sd)
            feats = np.concatenate((feats, x), axis=0)
            try:
                bounds.append(x.shape[0] + bounds[-1])
            except:
                bounds.append(x.shape[0])

    
    plt.imshow(feats, interpolation="nearest", aspect="auto")
    for bound in bounds:
        plt.axhline(bound, color="magenta", linewidth=2.0)
    plt.show()

    f = open(output, 'w')
    cPickle.dump(X, f)
    f.close()


def compute_N_cliques(maindir, cliques, N=10, output="cliques.pk"):
    """Computes the features for N cliques."""
    X = []
    clique_ids = []
    for i in xrange(N):
        clique_id = random.randint(0, len(cliques)-1)
        while clique_id in clique_ids:
            clique_id = random.randint(0, len(cliques)-1)
        clique_ids.append(clique_id)
        x =[]
        for tid in cliques[clique_id]:
            path = utils.path_from_tid(maindir, tid)
            feats = utils.extract_feats(path)
            if feats == None:
                continue
            x.append(feats)
        X.append(x)
    
    f = open(output, 'w')
    cPickle.dump(X, f)
    f.close()

def main():
    # Args parser
    parser = argparse.ArgumentParser(description=
                "Evaluates the average rank and mean AP for the SHS")
    parser.add_argument("maindir", action="store",
                        help="/data directory")
    parser.add_argument("shsf", action="store",
                        help="SHS dataset files")

    args = parser.parse_args()

    # sanity cheks
    assert os.path.isdir(args.maindir)
    assert os.path.isfile(args.shsf)

    # read cliques and all tracks
    cliques, all_tracks = utils.read_shs_file(args.shsf)

    # Compute specific features
    #compute_one_clique(args.maindir, cliques, clique_id=3753, mu=dict["mu"], 
    #                    sd=dict["sd"])
    
    #print "Computing original features"
    #compute_original_feats(args.maindir, all_tracks, cliques)

    print "Computing clique indeces"
    compute_clique_idxs(cliques, all_tracks)

    #print "Computing N cliques"
    #compute_N_cliques(args.maindir, cliques, N=10)

    # done
    print 'DONE!'

if __name__ == '__main__':
    main()