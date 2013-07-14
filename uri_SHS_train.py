#!/usr/bin/env python
"""
Test with meanAP and average rank
"""


import os
import sys
import cPickle
import pickle
import numpy as np
import argparse
from scipy.spatial import distance
import time
# local stuff
import pca
import hdf5_getters as GETTERS
import dan_tools
import time
import utils
import scipy.cluster.vq as vq
import pylab as plt
from transforms import load_transform
import analyze_stats as anst

# params, for ICMR paper: 75 and 1.96
WIN = 75
PATCH_LEN = WIN*12

def compute_feats(track_ids, maindir, d, lda_file=None, lda_n=0, codes=None, 
        ver=True):
    """Computes the features using the dictionary d. If it doesn't exist, 
     computes them using TBM method."""
    if d != "":
        fx = load_transform(d)
        K = int(d.split("_")[1].split("E")[1])
    else:
        K = PATCH_LEN
    
    if codes == None:
        compute_codes = True
        codes = np.ones((len(track_ids),K)) * np.nan
    else:
        compute_codes = False
    if lda_file != None:
        if lda_n == 0: n_comp = 50
        elif lda_n == 1: n_comp = 100
        elif lda_n == 2: n_comp = 200
    else:
        n_comp = 900

    # DrLIM / PCAstuff!   
    #n_comp = 200
    #fx_drlim = load_transform("DrLIMTransform_200.pk")
    #pca = utils.load_pickle("pca" + str(n_comp) + "_90000.pk")
    #lda200 = utils.load_pickle("lda200_24000.pk")

    n_comp = lda_file[0].n_components

    final_feats = np.ones((codes.shape[0],n_comp)) * np.nan
    for cnt, tid in enumerate(track_ids):
        if compute_codes:
            path = utils.path_from_tid(maindir, tid)
            feats = utils.extract_feats(path)
            if feats == None:
                continue
            if d != "":
                H = fx(feats)
            else:
                H = feats
            H = np.median(H, axis=0)
        else:
            H = codes[cnt]

        if compute_codes:
            codes[cnt] = H.copy()

        # Apply LDA if needed
        if lda_file != None:
            H = lda_file[lda_n].transform(H)

        #DrLim
        #H = fx_drlim(H).T
        #H = pca.transform(H)
        #H = lda200.transform(H)

        final_feats[cnt] = dan_tools.chromnorm(H.reshape(H.shape[0], 1)).squeeze()

        if ver:
            if cnt % 50 == 1:
                print "----Computing features %.1f%%" % (cnt/float(len(track_ids)) * 100)

    # Save codes
    if compute_codes:
        f = open("codes-" + os.path.basename(d) + ".pk", "w")
        cPickle.dump(codes, f, protocol=1)
        f.close()

    # Save features
    f = open("feats-" + os.path.basename(d) + ".pk", "w")
    cPickle.dump(final_feats, f, protocol=1)
    f.close()

    print "Features Computed"
    return final_feats

def load_pickle(file):
    """Gets the file from the cPickle file dictfile."""
    f = open(file, 'r')
    dict = cPickle.load(f)
    f.close()
    print "file %s loaded" % file
    return dict

def score(feats, clique_ids, lda_idx=0, stats_len=None, ver=True):
    #stats = np.zeros(5236)
    if stats_len is None:
        stats = [np.inf]*len(feats)
    else:
        stats = [np.inf]*stats_len
    
    # For each track id that has a clique id
    q = 0
    for i, clique_id in enumerate(clique_ids):
        if clique_id == -1:
            continue
        D = distance.cdist(feats[i][np.newaxis,:], feats, metric="euclidean")
        s = np.argsort(D)[0]
        sorted_cliques = clique_ids[s]
        r = np.argwhere( sorted_cliques == clique_id )[1:]
        if len(r) > 0:
            stats[i] = r
        q += 1
        if ver:
            if q % 400 == 0:
                print 'After %d queries: average rank per track: %.2f, clique: %.2f, MAP: %.5f' \
                    % (q, anst.average_rank_per_track(stats),
                        anst.average_rank_per_clique(stats),
                        anst.mean_average_precision(stats, n=len(feats)))

    return stats

def main():
    # Args parser
    parser = argparse.ArgumentParser(description=
                "Evaluates the average rank and mean AP for the SHS")
    parser.add_argument("-dictfile", action="store", default="",
                        help="Pickle to the learned dictionary")
    parser.add_argument("-lda", action="store", nargs=2, default=[None,0], 
                        help="LDA file and version", metavar=('lda.pkl', 'n'))
    parser.add_argument("-codes", action="store", default=None, dest="codesfile",
                        help="Pickle to the features file")
    parser.add_argument("-f", action="store", default="", dest="featfile",
                        help="Pickle to the final features")

    args = parser.parse_args()
    start_time = time.time()
    maindir = "SHSTrain/"
    shsf = "SHS/shs_dataset_train.txt"

    # sanity cheks
    utils.assert_file(args.dictfile)
    utils.assert_file(maindir)
    utils.assert_file(shsf)

    # read clique ids and track ids
    cliques, all_tracks = utils.read_shs_file(shsf)
    track_ids = all_tracks.keys()
    clique_ids = np.asarray(utils.compute_clique_idxs(track_ids, cliques))
    print "Track ids and clique ids read"

    # read LDA file
    lda_file = args.lda[0]
    if lda_file != None:
        lda_file = load_pickle(lda_file)
        print "LDA file read"

    # read codes file
    codesfile = args.codesfile
    if codesfile != None:
        codesfile = load_pickle(codesfile)
        print "Codes file read"

    # Compute features if needed
    if args.featfile == "":
        feats = compute_feats(track_ids, maindir, args.dictfile,
            lda_file=lda_file, lda_n=int(args.lda[1]), codes=codesfile)
    else:  
        feats = load_pickle(args.featfile)

    # Scores
    feats, clique_ids, track_ids = utils.clean_feats(feats, clique_ids, track_ids)
    stats = score(feats, clique_ids)

    f = open("stats-" + os.path.basename(args.dictfile), "w")
    cPickle.dump(stats, f, protocol=1)
    f.close()

    # done
    print 'DONE!'
    print 'Average rank per track: %.2f, clique: %.2f, MAP: %.5f' \
                % (anst.average_rank_per_track(stats),
                    anst.average_rank_per_clique(stats),
                    anst.mean_average_precision(stats, n=len(feats)))
    print "Took %.2f seconds" % (time.time() - start_time)

if __name__ == '__main__':
    main()
