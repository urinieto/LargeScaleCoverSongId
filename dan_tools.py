"""
Translation of Dan's MATLAB tools
"""

import copy
import numpy as np
import scipy.fftpack
import scipy.signal

import hdf5_getters as GETTERS


def L1norm(F):
    """divide over the sum of the absolute values."""
    return F/np.sum(np.abs(F))

def chromnorm(F, P=2.):
    """
    N = chromnorm(F,P)
       Normalize each column of a chroma ftrvec to unit norm
       so cross-correlation will give cosine distance
       S returns the per-column original norms, for reconstruction
       P is optional exponent for the norm, default 2.
    2006-07-14 dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    """
    nchr, nbts = F.shape
    if not np.isinf(P):
        S = np.power(np.sum(np.power(F,P), axis=0),(1./P));
    else:
        S = F.max();
    return F/S


def chrompwr(X, P=.5):
    """
    Y = chrompwr(X,P)  raise chroma columns to a power, preserving norm
    2006-07-12 dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    """
    nchr, nbts = X.shape
    # norms of each input col
    CMn = np.tile(np.sqrt(np.sum(X * X, axis=0)), (nchr, 1))
    CMn[np.where(CMn==0)] = 1
    # normalize each input col, raise to power
    CMp = np.power(X/CMn, P)
    # norms of each resulant column
    CMpn = np.tile(np.sqrt(np.sum(CMp * CMp, axis=0)), (nchr, 1))
    CMpn[np.where(CMpn==0)] = 1.
    # rescale cols so norm of output cols match norms of input cols
    return CMn * (CMp / CMpn)


def chromhpf(F, alpha=.9):
    """
    G = chromhpf(F,alpha)  high-pass filter a chroma matrix
    F is a chroma matrix (12 rows x N time steps)
    Apply a one-pole, one-zero high pass filter to each
    row, with a pole at alpha (0..1, default 0.99)
    2007-06-17 Dan Ellis dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    """
    nr, nc = F.shape
    G = np.zeros((nr, nc))
    for i in range(nr):
        G[i,:] = scipy.signal.lfilter([1,-1],[1,-alpha], F[i,:])
    return G


def bttonnetz_to_fftmat(bttonnetz, win=75):
    """
    Stack the flattened result of fft2 on patches 12 x win
    Translation of my own matlab function
    -> python: TBM, 2011-11-05, TESTED
    """
    # 12 semitones
    nchrm, nbeats = bttonnetz.shape
    assert nchrm == 6, 'beat-aligned matrix transposed?'
    if nbeats < win:
        return None
    # output
    fftmat = np.zeros((nchrm * win, nbeats - win + 1))
    for i in range(nbeats-win+1):
        patch = fftshift(magnitude(fft2(bttonnetz[:,i:i+win])))
        # 'F' to copy Matlab, otherwise 'C'
        fftmat[:, i] = patch.flatten('F')
    return fftmat


def btchroma_to_fftmat(btchroma, win=75):
    """
    Stack the flattened result of fft2 on patches 12 x win
    Translation of my own matlab function
    -> python: TBM, 2011-11-05, TESTED
    """
    # 12 semitones
    nchrm, nbeats = btchroma.shape
    assert nchrm == 12, 'beat-aligned matrix transposed?'
    if nbeats < win:
        return None
    # output
    fftmat = np.zeros((nchrm * win, nbeats - win + 1))
    for i in range(nbeats-win+1):
        patch = fftshift(magnitude(fft2(btchroma[:,i:i+win])))
        # 'F' to copy Matlab, otherwise 'C'
        fftmat[:, i] = patch.flatten('F')
    return fftmat


def fft2(X):
    """
    Same as fft2 in Matlab
    -> python: TBM, 2011-11-05, TESTED
    ok, useless, but needed to be sure it was doing the same thing
    """
    return scipy.fftpack.fft2(X)


def fftshift(X):
    """
    Same as fftshift in Matlab
    -> python: TBM, 2011-11-05, TESTED
    ok, useless, but needed to be sure it was doing the same thing
    """
    return scipy.fftpack.fftshift(X)


def magnitude(X):
    """
    Magnitude of a complex matrix
    """
    r = np.real(X)
    i = np.imag(X)
    return np.sqrt(r * r + i * i);


def msd_beatchroma(filename):
    """
    Get the same beatchroma as Dan
    Our filename is the full path
    TESTED
    """
    nchr=12
    # get segments, pitches, beats, loudness
    h5 = GETTERS.open_h5_file_read(filename)
    pitches = GETTERS.get_segments_pitches(h5).T
    loudness = GETTERS.get_segments_loudness_start(h5)
    Tsegs = GETTERS.get_segments_start(h5)
    Tbeats = GETTERS.get_beats_start(h5)
    h5.close()
    # sanity checks
    if len(Tsegs) < 3 or len(Tbeats) < 2:
        return None
    # get chroma and apply per segments loudness
    Segs = pitches * np.tile(np.power(10., loudness/20.), (nchr, 1))
    if Segs.shape[0] < 12 or Segs.shape[1] < 3:
        return None
    # properly figure time overlaps and weights
    C = resample_mx(Segs, Tsegs, Tbeats)
    # renormalize columns
    n = C.max(axis=0)
    return C * np.tile(1./n, (nchr, 1))


def resample_mx(X, incolpos, outcolpos):
    """
    Y = resample_mx(X, incolpos, outcolpos)
    X is taken as a set of columns, each starting at 'time'
    colpos, and continuing until the start of the next column.
    Y is a similar matrix, with time boundaries defined by
    outcolpos.  Each column of Y is a duration-weighted average of
    the overlapping columns of X.
    2010-04-14 Dan Ellis dpwe@ee.columbia.edu  based on samplemx/beatavg
    -> python: TBM, 2011-11-05, TESTED
    """
    noutcols = len(outcolpos)
    Y = np.zeros((X.shape[0], noutcols))
    # assign 'end times' to final columns
    if outcolpos.max() > incolpos.max():
        incolpos = np.concatenate([incolpos,[outcolpos.max()]])
        X = np.concatenate([X, X[:,-1].reshape(X.shape[0],1)], axis=1)
    outcolpos = np.concatenate([outcolpos, [outcolpos[-1]]])
    # durations (default weights) of input columns)
    incoldurs = np.concatenate([np.diff(incolpos), [1]])

    for c in range(noutcols):
        firstincol = np.where(incolpos <= outcolpos[c])[0][-1]
        firstincolnext = np.where(incolpos < outcolpos[c+1])[0][-1]
        lastincol = max(firstincol,firstincolnext)
        # default weights
        wts = copy.deepcopy(incoldurs[firstincol:lastincol+1])
        # now fix up by partial overlap at ends
        if len(wts) > 1:
            wts[0] = wts[0] - (outcolpos[c] - incolpos[firstincol])
            wts[-1] = wts[-1] - (incolpos[lastincol+1] - outcolpos[c+1])
        wts = wts * 1. /sum(wts)
        Y[:,c] = np.dot(X[:,firstincol:lastincol+1], wts)
    # done
    return Y
