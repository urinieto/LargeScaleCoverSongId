#!/usr/bin/env python
"""
Binary task of cover song identification using the Millions Song Dataset 
and the Second Hand Song dataset.

It takes the Million Song Dataset path as an argument. 

The list of queries to test must be located in:
./SHS/list_500queries.txt

The training set of the Second Hand Song dataset must be located in:
./SHS/shs_dataset_train.txt

Please, read the README.md file for more info on how to run this code.

References:
Bertin-Mahieux, T., & Ellis, D. P. W. (2012). Large-Scale Cover Song 
Recognition Using The 2D Fourier Transform Magnitude. In Proc. of the 13th 
International Society for Music Information Retrieval Conference (pp. 241-246).
Porto, Portugal.

Humphrey, E. J., Nieto, O., & Bello, J. P. (2013). Data Driven and 
Discriminative Projections for Large-Scale Cover Song Identification. 
In Proc. of the 14th International Society for Music Information Retrieval 
Conference. Curitiba, Brazil.

Created by Thierry Bertin-Mahieux (tb2332@columbia.edu)
Modified by Uri Nieto (oriol@nyu.edu)

----
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
import sys
import time

# local stuff
import pca
import hdf5_getters as GETTERS
import dan_tools
import utils
from transforms import load_transform

# Thierry's original parameters for ISMIR paper
WIN = 75
PWR = 1.96
PATCH_LEN = WIN*12

# Set up logger
logger = utils.configure_logger()

def extract_feats(filename, d, lda_file=None, lda_n=0, ver=True, fx=None):
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
    if d != "" and fx is not None:
        fx = load_transform(d)
    
    # 1.- Beat Synchronous Chroma
    # 2.- L2-Norm
    # 3.- Shingle (PATCH_LEN: 75 x 12)
    # 4.- 2D-FFT
    feats = utils.extract_feats(filename)
    if feats is None:
        return None

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

    # Apply LDA if needed
    if lda_file is not None:
        # 10.- Dimensionality Reduction
        H = lda_file[lda_n].transform(H)

    # 11.- L2-Norm
    feats = dan_tools.chromnorm(H.reshape(H.shape[0], 1)).squeeze()

    return feats


def read_query_file(queriesf):
    """Read queries, return triplets (query/good/bad)."""
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
    logger.info('Found %d queries from file %s' % (len(queries), queriesf))
    return queries


def main():
    # Args parser
    parser = argparse.ArgumentParser(description=
                "Evaluates the 500 binary queries from the SHS data set",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("msd_dir", action="store",
                        help="Million Song Dataset main directory")
    parser.add_argument("-dictfile", action="store", default="",
                        help="Pickle to the learned dictionary")
    parser.add_argument("-lda", action="store", nargs=2, default=[None,0], 
                        help="LDA file and version", metavar=('lda.pkl', 'n'))
    parser.add_argument("-pca", nargs=2, metavar=('f.pkl', 'n'), 
                        default=("", 0),
                        help="pca model saved in a pickle file, " \
                        "use n dimensions")
    # Parse
    args = parser.parse_args()

    # Track time
    start_time = time.time()

    maindir = args.msd_dir
    queriesf = "SHS/list_500queries.txt"
    shsf = "SHS/shs_dataset_train.txt"
    lda = args.lda[0]
    lda_n = int(args.lda[1])
    pcafile = args.pca[0]
    pcadim = int(args.pca[1])

    # sanity cheks
    utils.assert_file(maindir)
    utils.assert_file(queriesf)
    utils.assert_file(shsf)
    utils.assert_file(pcafile)

    # read queries
    queries = read_query_file(queriesf)

    # load pca
    trainedpca = None
    if pcafile != "":
        f = open(pcafile, 'r')
        trainedpca = cPickle.load(f)
        f.close()
        assert pcadim > 0
        logger.info('trained pca loaded')

    # load lda
    if lda != None:
        lda = utils.load_pickle(lda)

    # to keep stats
    results = []

    # iterate over queries
    logger.info("Starting the binary task...")
    for triplet in queries:
        # get features
        filenames = map(lambda tid: utils.path_from_tid(maindir, tid), triplet)
        triplet_feats = map(lambda f: extract_feats(f, args.dictfile, 
                                    lda_file=lda, lda_n=lda_n), filenames)
        if None in triplet_feats:
            continue

        # Apply pca if needed
        if trainedpca:
            triplet_feats = map(lambda feat: \
                                trainedpca.apply_newdata(feat, ndims=pcadim),
                                triplet_feats)
            assert triplet_feats[np.random.randint(3)].shape[0] == pcadim
        
        # Compute result
        res1 = triplet_feats[0] - triplet_feats[1]
        res1 = np.sum(res1 * res1)
        res2 = triplet_feats[0] - triplet_feats[2]
        res2 = np.sum(res2 * res2)
        if res1 < res2:
            results.append(1)
        else:
            results.append(0)

        # verbose
        if len(results) % 5 == 0:
            logger.info(' --- after %d queries, accuracy: %.1f %%' % \
                            (len(results), 100. * np.mean(results)))
    # done
    logger.info('After %d queries, accuracy: %.1f %%' % (len(results),
                                                100. * np.mean(results)))
    logger.info('Done! Took %.2f seconds' % (time.time() - start_time))

if __name__ == '__main__':
    main()
