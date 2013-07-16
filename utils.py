#!/usr/bin/env python

"""utilps.py: Handy functions for the summary project."""
__author__      = "Uri Nieto"
__date__        = "24/10/12"

import cPickle
import glob
import logging
import numpy as np
import os
import pylab as plt
from sklearn.lda import LDA

# local files
import dan_tools
import analyze_stats as anst

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

def fit_LDA_from_codes_file(codes_file):
    """Fits and LDA from a codes file and saves it into a new pickle file."""
    clique_idx = load_pickle("clique_idx.pk")
    codes = load_pickle(codes_file)

    # Remove nans
    nan_idx = np.unique(np.where(np.isnan(codes))[0])
    codes = np.delete(codes, nan_idx, axis=0)
    clique_idx = np.delete(clique_idx, nan_idx, axis=0)

    res = []
    components = [50,100,200]
    for c in components:
        lda = LDA(n_components=c)
        lda.fit(codes, clique_idx)
        res.append(lda)
    f = open(codes_file.strip(".pk") + "_LDAs.pk", "w")
    cPickle.dump(res, f, protocol=1)
    f.close()

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

def clean_feats(feats, clique_ids, track_ids):
    """Removes any nan feats from the input parameters."""
    nan_idx = np.unique(np.where(np.isnan(feats))[0])
    feats = np.delete(feats, nan_idx, axis=0)
    clique_ids = np.delete(clique_ids, nan_idx, axis=0)
    track_ids = np.delete(track_ids, nan_idx, axis=0)
    return feats, clique_ids, track_ids

def clean_feats2(feats, clique_ids):
    """Removes any nan feats from the input parameters."""
    nan_idx = np.unique(np.where(np.isnan(feats))[0])
    feats = np.delete(feats, nan_idx, axis=0)
    clique_ids = np.delete(clique_ids, nan_idx, axis=0)
    return feats, clique_ids

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

def lda_chart():
    import uri_SHS_train as U
    
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



