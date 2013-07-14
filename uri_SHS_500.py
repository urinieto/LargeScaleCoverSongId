#!/usr/bin/env python
"""
Quickly test cover song on the usual list of 500 binary queries
"""


import os
import sys
import cPickle
import numpy as np
import argparse
# local stuff
import pca
import hdf5_getters as GETTERS
import dan_tools

import utils


def extract_feats(filename, feats, track_ids, lda=None):
    """
    Finds the filename inside track_id and returns its correspondent features
    from feats
    """

    try:
        feat = feats[track_ids.index(filename)]
        if lda != None:
            feat = lda[0].transform(feat)
        feat = dan_tools.chromnorm(feat.reshape(feat.shape[0], 1)).squeeze()
        return feat
    except:
        return None


def read_query_file(queriesf):
    """
    Read queries, return triplets (query/good/bad)
    """
    queries = []
    triplet = []
    f = open(queriesf, 'r')
    for line in f.xreadlines():
        if line == '' or line.strip() == '':
            continue
        if line[0] == '#':
            continue
        if line[0] == '%':
            assert len(triplet) == 0 or len(triplet) == 3
            if len(triplet) > 0:
                queries.append(triplet)
                triplet = []
            continue
        tid = line.strip()
        assert len(tid) == 18 and tid[:2] == 'TR'
        triplet.append(tid)
    assert len(triplet) == 3
    queries.append(triplet)
    f.close()
    print 'Found %d queries from file %s' % (len(queries), queriesf)
    return queries


def main():
    # Args parser
    parser = argparse.ArgumentParser(description=
                "Evaluates the 500 binary queries from the SHS data set")
    parser.add_argument("list500queries", action="store",
                        help="usual queries from our cover papers")
    parser.add_argument("-f", action="store", dest="features",
                        help="cPickle file containing the features.")
    parser.add_argument("-lda", action="store", dest="lda", default=None,
                        help="cPickle file containing the LDA transform.")
    # Parse
    args = parser.parse_args()
    maindir = "SHSTrain/"
    queriesf = args.list500queries
    shsf = "SHS/shs_dataset_train.txt"
    lda = args.lda

    # sanity cheks
    assert os.path.isdir(maindir)
    assert os.path.isfile(queriesf)
    assert os.path.isfile(shsf)

    # read queries
    queries = read_query_file(queriesf)

    # read clique ids and track ids
    cliques, all_tracks = utils.read_shs_file(shsf)
    track_ids = all_tracks.keys()
    clique_ids = np.asarray(utils.compute_clique_idxs(track_ids, cliques))
    print "Track ids and clique ids read"

    # read track ids
    feats = utils.load_pickle(args.features)

    if lda != None:
        lda = utils.load_pickle(args.lda)

    # to keep stats
    cnt = 0
    cnt_good = 0

    # iterate over queries!
    for triplet in queries:
        # get features
        triplet_feats = map(lambda f: extract_feats(f, feats, track_ids, lda), 
                            triplet)
        if None in triplet_feats:
            continue
        # did we get it right?
        res1 = triplet_feats[0] - triplet_feats[1]
        res1 = np.sum(res1 * res1)
        res2 = triplet_feats[0] - triplet_feats[2]
        res2 = np.sum(res2 * res2)
        if res1 < res2:
            cnt_good += 1
        else:
            #print triplet
            pass
        # cnt
        cnt += 1
        if cnt >= 500:
            break
        # verbose
        if cnt % 50 == 0:
            print ' --- after %d queries, accuracy: %.3f' % (cnt,
                                                             100. * cnt_good / cnt)
    # done
    print 'DONE!'
    print 'After %d queries, accuracy: %.4f' % (cnt,
                                                100. * cnt_good / cnt)

if __name__ == '__main__':
    main()
