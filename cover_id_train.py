#!/usr/bin/env python
"""
This script computes the features necessary to achieve the results on the SHS
training set reported in the paper:

Humphrey, E. J., Nieto, O., & Bello, J. P. (2013). Data Driven and 
Discriminative Projections for Large-Scale Cover Song Identification. In Proc. 
of the 14th International Society for Music Information Retrieval Conference. 
Curitiba, Brazil.

A previously learned dictionary to convert the 2D-FMC features into codes clean_feats
be found in "models/BasisProjection2_ke2045_actEdot_shkE0x200_anormETrue.pk".

To use it, run the script as follows:
./cover_id_train.py -dictfile models/BasisProjection2_ke2045_actEdot_shkE0x200_anormETrue.pk

The PCA transform previously learned by Thierry can be found in:
"models/pca_250Kexamples_900dim_nocovers.pkl"

To use it, with an N number of dimensions, run the script as follows:
./cover_id_train.py -pca models/pca_250Kexamples_900dim_nocovers.pkl N

Th script saves the provisional codes in "results/codes-$DICTNAME$.pk". To learn
a LDA transform based on the codes, use the function "fit_LDA_from_codes_file"
in the utils.py file.

For more info, run:
./cover_id_train.py -h

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
     computes them using Thierry's method.

     The improved pipeline is composed of 11 steps:

        1.- Beat Synchronous Chroma
        2.- L2-Norm
        3.- Shingle (PATCH_LEN: 75 x 12)
        4.- 2D-FFT
        5.- L2-Norm
        6.- Log-Scale
        7.- Sparse Coding
        8.- Shrinkage
        9.- Median Aggregation
        10.- Dimensionality Reduction
        11.- L2-Norm

    Original method by Thierry doesn't include steps 5,6,7,8,11.
     """
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
        K = codes[0].shape[0]
    if lda_file is not None:
        if lda_n == 0: n_comp = 50
        elif lda_n == 1: n_comp = 100
        elif lda_n == 2: n_comp = 200
    else:
        n_comp = K 

    final_feats = np.ones((codes.shape[0],n_comp)) * np.nan
    orig_feats = []
    for cnt, tid in enumerate(track_ids):
        if compute_codes:
            path = utils.path_from_tid(maindir, tid)

            # 1.- Beat Synchronous Chroma
            # 2.- L2-Norm
            # 3.- Shingle (PATCH_LEN: 75 x 12)
            # 4.- 2D-FFT
            feats = utils.extract_feats(path)
            #orig_feats.append(feats)    # Store orig feats
            if feats == None:
                continue
            
            if d != "":
                # 5.- L2-Norm
                # 6.- Log-Scale
                # 7.- Sparse Coding
                # 8.- Shrinkage
                H = fx(feats)
            else:
                H = feats
            #. 9.- Median Aggregation
            H = np.median(H, axis=0)
        else:
            H = codes[cnt]

        if compute_codes:
            codes[cnt] = H.copy()

        # Apply LDA if needed
        if lda_file is not None:
            # 10.- Dimensionality Reduction
            H = lda_file[lda_n].transform(H)

        # 11.- L2-Norm
        final_feats[cnt] = dan_tools.chromnorm(H.reshape(H.shape[0], 1)).squeeze()

        if ver:
            if cnt % 50 == 1:
                logger.info("----Computing features %.1f%%" % \
                            (cnt/float(len(track_ids)) * 100))

    if d == "":
        d = "orig" # For saving purposes
    
    # Save codes
    utils.create_dir("results")
    if compute_codes:
        utils.save_pickle(codes, "results/codes-" + os.path.basename(d) + ".pk")

    # Save features
    #utils.save_pickle(orig_feats, "results/feats-" + os.path.basename(d) + ".pk")

    logger.info("Features Computed")
    return final_feats

def score(feats, clique_ids, lda_idx=0, stats_len=None, ver=True):
    """Compute the scores of the entire train dataset."""
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
                "Cover song ID on the training Second Hand Song dataset",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument("-pca", nargs=2, metavar=('f.pkl', 'n'), 
                        default=("", 0),
                        help="pca model saved in a pickle file, " \
                        "use n dimensions")

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
    utils.save_pickle(clique_ids, "SHS/clique_ids_train.pk")
    utils.save_pickle(track_ids, "SHS/track_ids_train.pk")

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

    # Apply PCA
    pcafile = args.pca[0]
    pcadim = int(args.pca[1])
    if pcafile != "":
        trainedpca = utils.load_pickle(pcafile)
        assert pcadim > 0
        logger.info('trained pca loaded')
        pcafeats = np.zeros((feats.shape[0], pcadim))
        for i,feat in enumerate(feats):
            pcafeats[i] = trainedpca.apply_newdata(feat, ndims=pcadim)
        feats = pcafeats

    # Scores
    feats, clique_ids, track_ids = utils.clean_feats(feats, clique_ids, track_ids)
    stats = score(feats, clique_ids)

    # Save data
    if dictfile == "":
        dictfile = "thierry" # For saving purposes
    utils.save_pickle(stats, "results/stats-" + os.path.basename(dictfile) + ".pk")

    # done
    logger.info('Average rank per track: %.2f, clique: %.2f, MAP: %.2f%%' \
                % (anst.average_rank_per_track(stats),
                    anst.average_rank_per_clique(stats),
                    anst.mean_average_precision(stats) * 100))
    logger.info("Done! Took %.2f seconds" % (time.time() - start_time))

if __name__ == '__main__':
    main()