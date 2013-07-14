'''
Created on Nov 26, 2012

@author: ejhumphrey
'''

import numpy as np

def ordered_recall_precision(x):
    """
    Compute recall-precision curves for a given document relevance
    vector.

    x : np.ndarray
        ordered list of binary documents, where True / 1 == hit
    """
    N_relevant = x.astype(int).sum()
    N_total = x.astype(int).sum()


    rel_count = np.arange(1,N_relevant+1)
    all_count = x.nonzero()[0].astype(float) + 1.0

    precision = rel_count / all_count
#    recall = np.linspace(0,1,N_relevant)
    recall = rel_count.astype(float)/N_relevant

    return recall,precision


def maxhold_recall_precision(recall, precision, eps=2.0**-10.0):
    """
    Max-hold precision curve.

    recall : np.ndarray
        monotonically increasing vector on the scale [0.0, 1.0]

    precision : np.ndarray
        bounded on [0.0, 1.0], same shape as 'recall'


    """
    N_relevant = len(recall)

    prec_max = [precision.max()]
    rec_max = [0.0]
    for n in range(1,N_relevant):
        p_n = precision[n:].max()
        if p_n < prec_max[-1]:
            # Double the r-points
            rec_max += [recall[n]-eps]
            prec_max += [prec_max[-1]]

        # add new ones
        rec_max += [recall[n]]
        prec_max += [p_n]

    return np.asarray(rec_max), np.asarray(prec_max)


def maxinterp_recall_precision(recall, precision, n_points, DISC=False, eps=2.0**-10.0):
    """
    Max-interp recall-precision curve with a given number of points.

    recall : np.ndarray
        monotonically increasing vector on the scale [0.0, 1.0]

    precision : np.ndarray
        bounded on [0.0, 1.0], same shape as 'recall'

    n_points : int
        number of evenly spaced recall values to interpolate

    DISC : bool [=False]
        toggle discontinuities for plotting

    eps : float
        tolerance for graphical precision at discontinuities

    """
    prec_out = [precision.max()]
    rec_out = [0.0]
    r_interp = np.linspace(0.0,1.0,n_points)
    for r in r_interp[1:-1]:
        n = (recall>r).argmax()
        p_r = precision[n:].max()
        if p_r < prec_out[-1] and DISC:
            # Double the r-points
            rec_out += [r-eps]
            prec_out += [prec_out[-1]]

        # add new ones
        rec_out += [r]
        prec_out += [p_r]

    # TODO: Final point is a hack, does not interpolate correctly
    rec_out += [1.0]
    prec_out += [precision[-1]]

    return np.asarray(rec_out), np.asarray(prec_out)

