#!/usr/bin/env python
"""


----
Authors: 
Uri Nieto (oriol@nyu.edu)
Eric J. Humphrey (ejhumphrey@nyu.edu)

----
License:
This code is distributed under the GNU LESSER PUBLIC LICENSE 
(LGPL, see www.gnu.org).

Copyright (c) 2012-2013 MARL@NYU.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of MARL, NYU nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""


import os
import sys
import cPickle
import pickle
import numpy as np
import argparse
from scipy.spatial import distance
import time
import glob

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

# Set up logger
logger = utils.configure_logger()

def compute_codes_orig_it(track_ids, maindir, clique_ids, start_idx, end_idx):
    """Computes the original features, based on Thierry and Ellis, 2012.
    Dimensionality reduction using PCA of 50, 100, and 200 components."""
    res = []
    trainedpca = utils.load_pickle("models/pca_250Kexamples_900dim_nocovers.pkl")
    pca_components = [50,100,200]

    # Init codes
    codes = []
    for n_comp in pca_components:
        codes.append(np.ones((end_idx-start_idx,n_comp)) * np.nan)

    for i, tid in enumerate(track_ids[start_idx:end_idx]):
        path = utils.path_from_tid(maindir, tid)
        feats = utils.extract_feats(path)
        if feats == None:
            continue
        med = np.median(feats, axis=0)
        for pca_idx, n_comp in enumerate(pca_components):
            tmp = dan_tools.chromnorm(med.reshape(med.shape[0], 
                                    1)).squeeze()
            codes[pca_idx][i] = trainedpca.apply_newdata(tmp, ndims=n_comp)
        if i % 1000 == 0:
            logger.info("Computed %d of %d track(s)" % (i, end_idx-start_idx))
    res = (codes, track_ids[start_idx:end_idx], clique_ids[start_idx:end_idx])
    return res

def compute_codes_it(track_ids, maindir, d, clique_ids, lda, 
        start_idx, end_idx):
    """Computes the features based on Humphrey, Nieto and Bello, 2013.
    Dimensionality reduction using LDA of 50, 100, and 200 components."""
    fx = load_transform(d)
    res = []
    lda_components = [50,100,200]

    # Init codes
    codes = []
    for n_comp in lda_components:
        codes.append(np.ones((end_idx-start_idx,n_comp)) * np.nan)

    for i, tid in enumerate(track_ids[start_idx:end_idx]):
        path = utils.path_from_tid(maindir, tid)
        feats = utils.extract_feats(path)
        if feats == None:
            continue
        med = np.median(fx(feats), axis=0)
        for lda_idx, n_comp in enumerate(lda_components):
            tmp = lda[lda_idx].transform(med)
            codes[lda_idx][i] = dan_tools.chromnorm(tmp.reshape(tmp.shape[0], 
                                    1)).squeeze()
        if i % 1000 == 0:
            logger.info("Computed %d of %d track(s)" % (i, end_idx-start_idx))
    res = (codes, track_ids[start_idx:end_idx], clique_ids[start_idx:end_idx])
    return res

def compute_codes(track_ids, maindir, d, N, clique_ids, lda):
    """Computes maximum 10,000 x 10 tracks. N is the index in the MSD:
        e.g. 
            if N = 1: tracks computed: from 100,000 to 199,999
            if N = 5: tracks computed: from 500,000 to 599,999
    """
    MAX     = 1e5 / 2
    ITER    = 1e4 / 2
    for it in xrange(10):
        logger.info("Computing %d of 10 iteration" % it)
        start_idx = int(N*MAX + it*ITER)
        end_idx = int(start_idx + ITER)
        codes = []
        strN = str(N)
        if N < 10:
            strN = "0" + str(N)
        if lda is None:
            codes = compute_codes_orig_it(track_ids, maindir, clique_ids,
                start_idx, end_idx)
            out_file = "msd_codes_orig/" + strN + str(it) + "-msd-codes.pk"
        else:
            codes = compute_codes_it(track_ids, maindir, d, clique_ids, lda,
                start_idx, end_idx)
            out_file = "msd_codes_bona/" + strN + str(it) + "-msd-codes.pk"
        f = open(out_file, "w")
        cPickle.dump(codes, f, protocol=1)
        f.close()

def score(feats, clique_ids, lda_idx=0):
    stats = [np.inf] * 5236
    #stats = [np.inf] * 12960
    #stats = [np.inf]*len(feats)
    
    # For each track id that has a clique id
    logger.info("Computing scores for the MSD...")
    q = 0
    for i, clique_id in enumerate(clique_ids):
        if clique_id == -1:
            continue
        D = distance.cdist(feats[i][np.newaxis,:], feats, metric="euclidean")
        s = np.argsort(D)[0]
        sorted_cliques = clique_ids[s]
        r = np.argwhere( sorted_cliques == clique_id )[1:]
        if len(r) > 0:
            stats[q] = r
        q += 1
        if q % 400 == 0:
            logger.info('After %d queries: average rank per track: %.2f'
                ', clique: %.2f, MAP: %.5f' \
                % (q, anst.average_rank_per_track(stats),
                    anst.average_rank_per_clique(stats),
                    anst.mean_average_precision(stats)))

    return stats


def main():
    # Args parser
    parser = argparse.ArgumentParser(description=
                "Evaluates the average rank and mean AP for the test SHS " \
                "over the entire MSD",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("msd_dir", action="store",
                        help="Million Song Dataset main directory")
    parser.add_argument("-dictfile", action="store", default="",
                        help="Pickle to the learned dictionary")
    parser.add_argument("-N", action="store", type=int, default=0,
                        help="Set of 100,000ths to be computed")
    parser.add_argument("-lda", action="store", default=None, 
                        help="LDA file")
    parser.add_argument("-codes", action="store", nargs=2, default=[None,0], 
                        dest="codesdir", metavar=("msd_codes_orig/", "n"),
                        help="Path to the folder with all the codes and "
                            "version to evaluate")

    args = parser.parse_args()
    start_time = time.time()
    maindir = arg.msd_dir
    shsf = "SHS/shs_dataset_test.txt"

    # sanity cheks
    utils.assert_file(maindir)
    utils.assert_file(shsf)

    # TODO: Not needed? Do as in train file.
    # read cliques and all tracks
    cliques, all_tracks = utils.read_shs_file(shsf)
    track_ids = utils.load_pickle("SHS/track_ids_test.pk")
    clique_ids = utils.load_pickle("SHS/clique_ids_test.pk")

    # read codes file
    codesdir = args.codesdir[0]
    if codesdir is not None:
        c = utils.load_pickle(codesdir)
        feats = c[0]
        logger.info("Codes files read")
    else:
        # read LDA file
        lda_file = args.lda
        if lda_file is not None:
            lda_file = utils.load_pickle(lda_file)
            logger.info("LDA file read") 

        utils.assert_file(args.dictfile)
        compute_codes(track_ids, maindir, args.dictfile, args.N, clique_ids, 
            lda_file)
        logger.info("Codes computation done!")
        logger.info("Took %.2f seconds" % (time.time() - start_time))
        sys.exit()

    # Scores
    feats, clique_ids, track_ids = utils.clean_feats(feats, clique_ids, track_ids)
    stats = score(feats, clique_ids)

    f = open("stats-test-kE2045-LDA50.pk" + os.path.basename(args.dictfile), "w")
    cPickle.dump(stats, f, protocol=1)
    f.close()

    # done
    logger.info('Average rank per track: %.2f, clique: %.2f, MAP: %.5f' \
                % (anst.average_rank_per_track(stats),
                    anst.average_rank_per_clique(stats),
                    anst.mean_average_precision(stats)))
    logger.info("Done! Took %.2f seconds" % (time.time() - start_time))

if __name__ == '__main__':
    main()
