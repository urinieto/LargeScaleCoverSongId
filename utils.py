"""
Utils function for the Large Scale Cover Song ID project.

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

import cPickle
import glob
import logging
import numpy as np
import os
import pylab as plt
import dan_tools
from sklearn.lda import LDA
from sklearn.decomposition import PCA

# local files
import analyze_stats as anst
from transforms import load_transform

### Logging methods
def configure_logger():
    """Configures the logger for this project."""
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s:%(lineno)d  %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    return get_logger()

def get_logger():
    """Returns the logger created for this project."""
    return logging.getLogger('coverSongs')

def chroma_to_tonnetz( C ):
    """Transforms chromagram to Tonnetz (Harte, Sandler, 2006)."""
    N = C.shape[0]
    T = np.zeros((N,6))

    r1 = 1      # Fifths
    r2 = 1      # Minor
    r3 = 0.5    # Major

    # Generate Transformation matrix
    phi = np.zeros((6,12))
    for i in range(6):
        for j in range(12):
            if i % 2 == 0:
                fun = np.sin
            else:
                fun = np.cos

            if i < 2:
                phi[i,j] = r1 * fun( j * 7*np.pi / 6. )
            elif i >= 2 and i < 4:
                phi[i,j] = r2 * fun( j * 3*np.pi / 2. )
            else:
                phi[i,j] = r3 * fun( j * 2*np.pi / 3. )

    # Do the transform to tonnetz
    for i in range(N):
        for d in range(6):
            T[i,d] = 1/float(C[i,:].sum()) * (phi[d,:] * C[i,:]).sum()

    return T


def create_dir(dir):
    """Creates a directory if it doesn't exist yet."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def merge_pitches( start, end, w, pitches ):
    """Merges the pitches[start:end+1] to a single segment in s.
    The pitches[end] has a weight of w.
    """
    merged = np.concatenate((pitches[start:end,:],
                             pitches[end,:][np.newaxis,:]*w)).mean(axis=0)
    return merged


def plot_features( feat ):
    """Plots the feature feat. Rows are time frames, columns are feat
    dimensions."""
    plt.figure()
    plt.imshow( np.transpose(feat), interpolation="nearest", aspect="auto" )


def compute_fft2d_feats(feats):
    """Gets the 2dFFT magnitude of the list of features."""
    fft_feats = []
    for feat in feats:
        fft_feats.append(np.abs(np.fft.fft2(feat)).flatten())
    return fft_feats


def compute_track_from_clique_dict(gt):
    track_dict = dict()
    for key in gt.keys():
        for track in gt[key]:
            track_dict[track] = key
    return track_dict


def read_shs_file(shsf):
    """Read shs file, return list of cliques and tracks."""
    sep = "<SEP>"
    cliques = []
    tracks = []
    curr_clique = None
    all_tracks = dict()
    f = open(shsf, 'r')
    for line in f.xreadlines():
        if line[0] == '%':
            if len(tracks) > 0:
                cliques.append(tracks)
                tracks = []
            curr_clique = line.split(',')[0][1:]
            continue
        if line[0] == 'T':
            tid = line.split(sep)[0]
            assert len(tid) == 18 and tid[:2] == 'TR'
            tracks.append(tid)
            all_tracks[tid] = [None, curr_clique]
    cliques.append(tracks)
    f.close()
    logger = get_logger()
    logger.info('Found %d cliques from file %s' % (len(cliques), shsf))
    return cliques, all_tracks


def assert_file(file):
    """Makes sure that the file exists."""
    if file != "":
        assert os.path.isfile(file) or os.path.isdir(file), \
            'ERROR: file %s does not exist' % file

def path_from_tid(maindir, tid):
    """Returns a full path based on a main directory and a track id."""
    p = os.path.join(maindir, tid[2])
    p = os.path.join(p, tid[3])
    p = os.path.join(p, tid[4])
    p = os.path.join(p, tid.upper() + '.h5')
    return p

def extract_feats(filename):
    """
    Return a all the patches for the data in the
    given file
    It uses 2D-FFT, etc
    Written by Thierry
    """
    PWR = 1.96
    WIN = 75

    # get btchroma
    feats = dan_tools.msd_beatchroma(filename)
    if feats is None:
        return None
    # apply pwr
    feats = dan_tools.chrompwr(feats, PWR)
    # extract fft
    feats = dan_tools.btchroma_to_fftmat(feats, WIN)
    if feats is None:
        return None
    # return the non-normalized features (L, 900)
    return feats.T

def extract_track_ids(maindir):
    """Extracts all the track ids from an MSD structure."""
    ext = ".h5"
    cnt = 0
    track_ids = []
    for root, dirs, files in os.walk(maindir):
        files = glob.glob(os.path.join(root, "*" + ext))
        for file in files:
            track_ids.append(os.path.basename(file).split(ext)[0])
        cnt += len(files)
    logger = get_logger()
    logger.info("Parsed %d files", cnt)
    return track_ids

def compute_clique_idxs(track_ids, cliques):
    """Returns an array of size len(track_ids) with the clique_id 
        for each track."""
    clique_ids = []
    logger = get_logger()
    logger.info("Computing clique indeces...")
    for cnt, tid in enumerate(track_ids):
        i = 0
        idx = -1
        for clique in cliques:
            if tid in clique:
                idx = i
                break
            i += 1
        clique_ids.append(idx)

        #if cnt % 50000 == 0:
        #    print "Iteration:", cnt
    return clique_ids

def clean_feats(feats, clique_ids, track_ids=[]):
    """Removes any nan feats from the input parameters."""
    nan_idx = np.unique(np.where(np.isnan(feats))[0])
    feats = np.delete(feats, nan_idx, axis=0)
    clique_ids = np.delete(clique_ids, nan_idx, axis=0)
    if track_ids == []:
        return feats, clique_ids
    track_ids = np.delete(track_ids, nan_idx, axis=0)
    return feats, clique_ids, track_ids

def load_pickle(file):
    """Gets the file from the cPickle file."""
    f = open(file, 'r')
    d = cPickle.load(f)
    f.close()
    logger = get_logger()
    logger.info("file %s loaded" % file)
    return d

def save_pickle(data, file):
    """Save the data into a cPickle file."""
    f = open(file, 'w')
    cPickle.dump(data, f, protocol=1)
    f.close()
    logger = get_logger()
    logger.info("file %s saved" % file)

def get_train_validation_sets(codes, cliques, tracks, N=9000):
    """Gets a training set and a validation set from a set of codes with
        corresponding cliques. N is the number of codes in the new 
        training set."""
    M = len(codes) - N
    codes_val = []
    cliques_val = []
    tracks_val = []
    n = 0
    idx = []
    chosen_idx = []
    m_idx = 0
    while m_idx < M:
        while len(np.where(cliques == cliques[n])[0]) <= 2:
            n += 1
        clique_ids = np.where(cliques == cliques[n])[0]
        n += 1
        for clique_id in clique_ids:
            if clique_id in chosen_idx:
                m_idx -= len(clique_ids)
                break
            codes_val.append(codes[clique_id])
            cliques_val.append(cliques[clique_id])
            tracks_val.append(tracks[clique_id])
            idx.append(clique_id)
            chosen_idx.append(clique_id)
        m_idx += len(clique_ids)

    codes_val = np.asarray(codes_val)
    cliques_val = np.asarray(cliques_val)
    idx = np.asarray(idx)
    codes_train = np.delete(codes, idx, axis=0)
    cliques_train = np.delete(cliques, idx, axis=0)
    tracks_train = np.delete(tracks, idx, axis=0)

    save_pickle(codes_val, "codes_val.pk")
    save_pickle(cliques_val, "cliques_val.pk")
    save_pickle(tracks_val, "tracks_val.pk")
    save_pickle(codes_train, "codes_t.pk")
    save_pickle(cliques_train, "cliques_t.pk")
    save_pickle(tracks_train, "tracks_t.pk")


def fit_PCA(maindir, d, origcodes_f="msd_codes_k2045", outpca="PCA-codes.pk", 
        N=50000, norm=False, pca_components=[50, 100, 200, 500, 1000]):
    """Fits a PCA transformation with N codes."""
    import binary_task as B
    import cover_id_test as CO

    logger = configure_logger()

    td = load_transform(d)
    codes = np.ones((N, 2045)) * np.nan
    k = 0

    # Load codes
    origcodes, track_ids, clique_ids = CO.load_codes(origcodes_f, -1, 30)

    while k < N:
        track_idx = np.random.random_integers(0,len(track_ids)-1)
        while  track_ids[track_idx] == -2:
            track_idx = np.random.random_integers(0,len(track_ids)-1)
        track_id = track_ids[track_idx]
        code = origcodes[track_idx]
        if code is not None:
            if norm:
                code = dan_tools.chromnorm(code.reshape(code.shape[0], 1)).squeeze()
            codes[k] = code
            # Marked as used
            track_ids[track_idx] = -2
            k += 1

        if k % 100 == 0:
            logger.info("----Computing features %.1f%%" % \
                            (k/float(N) * 100))

    # Remove nans
    nan_idx = np.unique(np.where(np.isnan(codes))[0])
    codes = np.delete(codes, nan_idx, axis=0)

    # Fit PCA
    res = []
    codes = np.asarray(codes)
    for c in pca_components:
        pca = PCA(n_components=c)
        pca.fit(codes)
        res.append(pca)

    # Save Result
    save_pickle(res, outpca)

def fit_LDA_from_codes_file(codes_file, clique_idx, lda_components=[50,100,200],
        outlda="LDAs.pk"):
    """Fits and LDA from a codes file and saves it into a new pickle file."""

    clique_idx = np.asarray(load_pickle(clique_idx))
    codes = np.asarray(load_pickle(codes_file))

    # Remove Nones
    none_idx = np.where(np.equal(codes, None))[0]
    codes = np.delete(codes, none_idx, axis=0)
    clique_idx = np.delete(clique_idx, none_idx, axis=0)
    # Hack to make it the right shape
    C = np.zeros((codes.shape[0], codes[0].shape[0]))
    k = 0
    for code in codes:
        C[k] = code
        k+=1
    codes = C

    # Remove nans
    nan_idx = np.unique(np.where(np.isnan(codes))[0])
    codes = np.delete(codes, nan_idx, axis=0)
    clique_idx = np.delete(clique_idx, nan_idx, axis=0)
    print codes.shape

    res = []
    for c in lda_components:
        lda = LDA(n_components=c)
        lda.fit(codes, clique_idx)
        res.append(lda)
    save_pickle(res, outlda)


def fit_LDA_filter(maindir, d, codes_f, N=9000, n=9, pca=None, pca_n=0, 
        norm=False, outlda="LDAs.pk", lda_components=[50,100,200]):
    """Fits an LDA with a filtered version of the dataset, such that each
        clique contains at least n tracks."""

    import cover_id_test as CO

    clique_test = load_pickle("SHS/clique_ids_test.pk")
    clique_train = load_pickle("SHS/clique_ids_train.pk")
    track_test = load_pickle("SHS/track_ids_test.pk")
    track_train = load_pickle("SHS/track_ids_train.pk")

    # Result to 
    codes = []
    labels = []

    if pca is not None:
        P = load_pickle(pca)
        P = P[pca_n]

    C = CO.load_codes(codes_f, -1, 30)
    C = C[0]

    # Load the codes from the training set
    codestrain = load_pickle("codes-shs-train-k2045.pk")

    clique_idx = 0
    label_id = 1000001

    td = load_transform(d)

    while len(codes) < N:
        # Pick the tracks from the train set that belong to a
        # clique that has at least n tracks
        if clique_idx < len(clique_train):
            while clique_idx < len(clique_train) and \
                    len(np.where(clique_train == clique_train[clique_idx])[0]) < n :
                clique_idx += 1

            if clique_idx < len(clique_train) and clique_train[clique_idx] != -2:
                for clique_id in \
                        np.where(clique_train == clique_train[clique_idx])[0]:
                    code = codestrain[clique_id]
                    if norm:
                        code = dan_tools.chromnorm(code.reshape(code.shape[0], 
                                        1)).squeeze()
                    clique_train[clique_id] = -2
                    if code is None:
                        continue
                    if pca is not None:
                        code = P.transform(code)
                    codes.append( code )
                    labels.append( clique_idx )

            clique_idx += 1

        # Pick random tracks from the MSD and assign new labels
        else:
            clique_id = np.random.random_integers(0, len(C)-1)
            while np.any(np.equal(C[clique_id], None)) or clique_test[clique_id] == -2:
                clique_id = np.random.random_integers(0, len(C)-1)
            code = C[clique_id]
            if norm:
                code = dan_tools.chromnorm(code.reshape(code.shape[0], 
                                        1)).squeeze()
            if pca is not None:
                code = P.transform(code)
            codes.append( code )
            labels.append( label_id )
            label_id += 1
            clique_test[clique_id] = -2

        print "Computed %d out of %d codes" % (len(codes), N)

    codes_pk = "codes_filter_LDA_PCA.pk"
    cliques_pk = "cliques_filter_LDA_PCA.pk"
    save_pickle(codes, codes_pk)
    save_pickle(labels, cliques_pk)

    # fit LDA and save model
    fit_LDA_from_codes_file(codes_pk, cliques_pk, lda_components, outlda)


def lda_chart():
    import cover_id_train as U
    
    codes = load_pickle("lda_tran_codes.pk")
    cliques = load_pickle("lda_train_clique_id.pk")
    tracks = load_pickle("lda_train_track_id.pk")

    codes_val = load_pickle("codes_val.pk")
    cliques_val = load_pickle("cliques_val.pk")
    tracks_val = load_pickle("tracks_val.pk")
    
    codes_train = load_pickle("codes_t.pk")
    cliques_train = load_pickle("cliques_t.pk")
    tracks_train = load_pickle("tracks_t.pk")

    codes, cliques, tracks = clean_feats(codes, cliques, tracks)
    codes_val, cliques_val, tracks_val = clean_feats(codes_val, cliques_val, tracks_val)
    codes_train, cliques_train, tracks_train = clean_feats(codes_train, cliques_train, tracks_train)

    codes_tot = np.concatenate((codes_val, codes_train), axis=0)
    cliques_tot = np.concatenate((cliques_val, [-1]*len(codes_train)), axis=0)    
    tracks_tot = list(tracks_val) + list(tracks_train)


    noises = np.arange(0,20000,500)
    n_comps = [50, 100, 200]
    maindir = "SHSTrain"
    d = "dicts/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk"
    logger = get_logger()
    for noise in noises:
        for n_comp in n_comps:
            # Fit LDA
            lda = LDA(n_components=n_comp)
            lda_codes = np.concatenate((codes_train, codes[13000:noise]), axis=0)
            lda_cliques = np.concatenate((cliques_train, cliques[13000:noise]), axis=0)
            lda = [lda.fit(lda_codes, lda_cliques)]

            # Compute features
            feats_train = U.compute_feats(tracks_train, maindir, d, lda_file=lda, 
                codes=codes_train, ver=False)
            stats_train = U.score(feats_train, cliques_train, ver=False)
            feats_tot = U.compute_feats(tracks_tot, maindir, d, lda_file=lda, 
                codes=codes_tot, ver=False)
            stats_val = U.score(feats_tot, cliques_tot, stats_len=len(cliques_val),
                ver=False)

            # Compute scores
            ar_train = anst.average_rank_per_track(stats_train)
            map_train = anst.mean_average_precision(stats_train, n=len(feats_train))
            ar_val = anst.average_rank_per_track(stats_val)
            map_val = anst.mean_average_precision(stats_val, n=len(feats_tot))

            # Print outs
            str_train = "lda:%d, dataset:%s, noise:%d, avg rank:%.2f, MAP:%.4f\n" % \
                (n_comp, "train", noise, ar_train, map_train)
            str_val = "lda:%d, dataset:%s, noise:%d, avg rank:%.2f, MAP:%.4f\n" % \
                (n_comp, "val", noise, ar_val, map_val)
            logger.info(str_train + str_val)

            f = open("lda_chart.txt", "a")
            f.write(str_train + str_val)
            f.close()

def compute_training_features(N=50000):
    """Computes N features for training purposes."""

    logger = configure_logger()
    maindir = "MSD"

    clique_test = load_pickle("SHS/clique_ids_test.pk")
    track_test = load_pickle("SHS/track_ids_test.pk")

    feats = []

    k = 0
    K = 1000 # Save file every K iterations
    while len(feats) + k*K < N:
        clique_id = np.random.random_integers(0,999999)
        while clique_test[clique_id] == -2:
            clique_id = np.random.random_integers(0,999999)
        track_id = track_test[clique_id]
        filename = path_from_tid(maindir, track_id)
        feat = extract_feats(filename)
        if feat is not None:
            feats.append((feat, clique_id, track_id))
            # Marked as used
            clique_test[clique_id] = -2

        if len(feats) % K == 0:
            save_pickle(feats, "feats_training_NE%d_kE%d.pk" % (N, k))
            feats = []
            k += 1

        if len(feats) % 100 == 0:
            logger.info("----Computing features %.1f%%" % \
                            (len(feats)/float(N) * 100 + k*K/float(N) * 100))

    save_pickle(feats, "feats_training_NE%d_kE%d.pk" % (N, k))

def compute_models():
    """Computes the different models for the MSD."""

    logger = configure_logger()
    maindir = "MSD"

    # Dictionary to use
    d = "models/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk"
    K = int(d.split("_")[1].split("E")[1])

    # PCA settings
    M = 150000
    pca_dims = [50,100,200,500,1000] # Number of dimensions (-1: no pca)
    norms = [True, False]

    # Compute the PCA models
    pca_files = []
    for norm in norms:
        out_pca_file = "models/PCAs_mE%d_normE%r_kE%d.pk" % \
            (M, norm, K)
        fit_PCA(maindir, d, origcodes_f="msd_codes_k2045", outpca=out_pca_file, 
                N=M, norm=norm, pca_components=pca_dims)
        pca_files.append(pca_files)

    # Append None to not apply PCA in one of the LDA models
    pca_files.append(None)

    # LDA settings
    Ns = [3000, 10000, 100000] # Number of tracks from training set + noise
    ns = [-1, 5, 10, 15] # Filtering: number of tracks per clique (-1: no filter)
    lda_dims = [50, 100, 200]

    # Compute the LDA models
    for pca_file in pca_files:
        for N in Ns:
            for n in ns:
                for norm in norms:
                    if pca_file is None:
                        out_lda_file = "models/LDAs_NE%d_nE%d_mE%d_normE%r_kE%d.pk" % \
                            (N, n, M, norm, K)
                        fit_LDA_filter(maindir, d, "msd_codes_k2045", N=N, n=n, 
                            norm=norm, outlda=out_lda_file, lda_components=lda_dims)
                    else:
                        for i in xrange(len(pca_dims)):
                            pca_dim = pca_dims[i]
                            out_lda_file = "models/LDAs_pcaE%d_NE%d_nE%d_mE%d_normE%r_kE%d.pk" % \
                                (N, pca_dim, n, M, norm, K)
                            fit_LDA_filter(maindir, d, "msd_codes_k2045", N=N, 
                                n=n, norm=norm, pca=pca_file, pca_n=i, 
                                outlda=out_lda_file, lda_components=lda_dims)

                    














