#!/usr/bin/env python
"""
Test with meanAP and average rank
"""

import argparse
import cPickle
import numpy as np
import os
import pickle
from scipy.spatial import distance
import sys
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

# Thierry's original parameters for ISMIR paper
WIN = 75
PWR = 1.96
PATCH_LEN = WIN*12

# Set up logger
logger = utils.configure_logger()

def compute_feats(track_ids, maindir, d, lda_file=None, lda_n=0, codes=None, 
        ver=True):
    """Computes the features using the dictionary d. If it doesn't exist, 
     computes them using Thierry's method."""
    if d != "":
        fx = load_transform(d)
        K = int(d.split("_")[1].split("E")[1])
    else:
        K = PATCH_LEN
    
    if codes is None:
        compute_codes = True
        codes = np.ones((len(track_ids),K)) * np.nan
    else:
        compute_codes = False
    if lda_file is not None:
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

    #n_comp = lda_file[0].n_components

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
        if lda_file is not None:
            H = lda_file[lda_n].transform(H)

        #DrLim
        #H = fx_drlim(H).T
        #H = pca.transform(H)
        #H = lda200.transform(H)

        final_feats[cnt] = dan_tools.chromnorm(H.reshape(H.shape[0], 1)).squeeze()

        if ver:
            if cnt % 50 == 1:
                logger.info("----Computing features %.1f%%" % \
                            (cnt/float(len(track_ids)) * 100))

    if d == "":
        d = "thierry" # For saving purposes
    
    # Save codes
    utils.create_dir("results")
    if compute_codes:
        utils.save_pickle(codes, "results/codes2-" + os.path.basename(d) + ".pk")

    # Save features
    utils.save_pickle(final_feats, "results/feats-" + os.path.basename(d) + ".pk")

    logger.info("Features Computed")
    return final_feats

def score(feats, clique_ids, lda_idx=0, stats_len=None, ver=True):
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
                logger.info('After %d queries: average rank per track: %.2f, '\
                    'clique: %.2f, MAP: %.5f' \
                    % (q, anst.average_rank_per_track(stats),
                        anst.average_rank_per_clique(stats),
                        anst.mean_average_precision(stats)))

    return stats

def main():
    # Args parser
    parser = argparse.ArgumentParser(description=
                "Cover song ID on the training Second Hand Song dataset")
    parser.add_argument("msd_dir", action="store",
                        help="Million Song Dataset main directory")
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
    maindir = args.msd_dir
    shsf = "SHS/shs_dataset_train.txt"
    dictfile = args.dictfile

    # sanity cheks
    utils.assert_file(dictfile)
    utils.assert_file(maindir)
    utils.assert_file(shsf)

    # read clique ids and track ids
    cliques, all_tracks = utils.read_shs_file(shsf)
    track_ids = all_tracks.keys()
    clique_ids = np.asarray(utils.compute_clique_idxs(track_ids, cliques))
    logger.info("Track ids and clique ids read")

    # read LDA file
    lda_file = args.lda[0]
    if lda_file != None:
        lda_file = utils.load_pickle(lda_file)
        logger.info("LDA file read")

    # read codes file
    codesfile = args.codesfile
    if codesfile != None:
        codesfile = utils.load_pickle(codesfile)
        logger.info("Codes file read")

    # Compute features if needed
    if args.featfile == "":
        feats = compute_feats(track_ids, maindir, dictfile,
            lda_file=lda_file, lda_n=int(args.lda[1]), codes=codesfile)
    else:  
        feats = utils.load_pickle(args.featfile)

    # Scores
    feats, clique_ids, track_ids = utils.clean_feats(feats, clique_ids, track_ids)
    stats = score(feats, clique_ids)

    # Save data
    if dictfile == "":
        dictfile = "thierry" # For saving purposes
    utils.save_pickle(stats, "results/stats-" + os.path.basename(dictfile) + ".pk")

    # done
    logger.info('Average rank per track: %.2f, clique: %.2f, MAP: %.5f' \
                % (anst.average_rank_per_track(stats),
                    anst.average_rank_per_clique(stats),
                    anst.mean_average_precision(stats)))
    logger.info("Done! Took %.2f seconds" % (time.time() - start_time))

if __name__ == '__main__':
    main()