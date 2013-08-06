#!/usr/bin/env python
"""Computes the result of a stats cPickle file.

A stats cPickle file has the following format:

- List of N elements, each representing a track.
- Each position (or track) contains the rank index of the covers
    corresponding to this position.

The results this script computes are:
- Mean Average Precision (MAP)
- Average Rank per track
- Average Rank per clique
- Precision at k (default k=10)

Plotting:
- Rank histograms (one or two stats files)

Created by Oriol Nieto (oriol@nyu.edu), 2013"""

import argparse
import cPickle
import numpy as np
import pylab as plt
import utils

def get_top_ranked(stats):
    tr = np.zeros(len(stats))
    for i,s in enumerate(stats):
        try:
            if not np.isnan(s[0]) or s[0] != np.inf:
                tr[i] = s[0]
        except:
            continue
    return tr

def get_average_rank(stats):
    tr = np.zeros(len(stats))
    for i,s in enumerate(stats):
        try:
            if not np.isnan(s[0]) or s[0] != np.inf:
                tr[i] = np.mean(s)
        except:
            continue
    return tr

def average_rank_per_track(stats):
    mean_r = []
    for s in stats:
        try:
            for rank in s:
                if not np.isnan(rank) or rank != np.inf:
                    mean_r.append(rank)
        except:
            continue
    return np.mean(mean_r)

def average_rank_per_clique(stats):
    mean_r = []
    for s in stats:
        try:
            mean_r.append(np.mean(s))
            if np.isnan(mean_r[-1]) or mean_r[-1] == np.inf:
                mean_r = mean_r[:-1]
        except:
            continue
    return np.mean(mean_r)

def precision_at_k(ranks, k):
    if k == 0: return 1.0
    ranks = np.asarray(ranks)
    relevant = len(np.where(ranks <= k)[0])
    return relevant / float(k)

def average_precision(stats, q, ver=False):
    try:
        nrel = len(stats[q]) # Number of relevant docs
    except:
        return np.nan
    ap = []
    for k in stats[q]:
        pk = precision_at_k(stats[q], k)
        ap.append(pk)
    return np.sum(ap) / float(nrel)

def average_precision_at_k(stats, k):
    precision = []
    for s in stats:
        precision.append(precision_at_k(s,k))
    return np.mean(precision)

def mean_average_precision(stats):
    Q = len(stats) # Number of queries
    ma_p = []
    for q in xrange(Q):
        ap = average_precision(stats, q)
        if np.isnan(ap):
            continue
        ma_p.append(ap)
    return np.mean(ma_p)

def mean_per_clique_count(stats, N=None):
    if N is None:
        N = len(stats)
    means = np.zeros(N)
    for n in xrange(1,N):
        m = []
        k = 0
        for s in stats:
            try:
                if len(s) == n:
                    k += 1
                    m.append(np.mean(s))
            except:
                continue
        if len(m) != 0:
            means[n] = np.mean(m)
    return means

##### PLOTTING

def compute_rank_histogram_buckets(stats):
    ranks = []
    for s in stats:
        try:
            for rank in s:
                ranks.append(rank)
        except:
            continue
    # Calculate histogram
    """
    hist = np.zeros(5) #1-10, 11-25, 26-50, 51-100, 101+
    for r in ranks:
        if r <= 10:
            hist[0] += 1
        elif r > 10 and r <= 25:
            hist[1] += 1
        elif r > 25 and r <= 50:
            hist[2] += 1
        elif r > 50 and r <= 100:
            hist[3] += 1
        elif r > 100:
            hist[4] += 1
    """
    hist = np.zeros(5) #1, 2, 3-5, 6-10, 11+
    for r in ranks:
        if r <= 1:
            hist[0] += 1
        elif r > 1 and r <= 2:
            hist[1] += 1
        elif r > 2 and r <= 5:
            hist[2] += 1
        elif r > 5 and r <= 10:
            hist[3] += 1
        elif r > 10:
            hist[4] += 1

    # Probability Density Function:
    hist = hist.astype(float)
    hist /= float(hist.sum())

    return hist

def plot_rank_histogram(stats, bins=5):
    hist = compute_rank_histogram_buckets(stats)

    # Plot histogram as PDF
    plt.bar(xrange(0,bins), hist, align="center")
    plt.title("Rank Histogram")
    plt.xlabel("Ranks")
    plt.ylabel("Normalized Count")
    plt.xticks(xrange(0,5), ("1-10", "11-25", "26-50", "51-100", "101+"))
    plt.show()

def plot_rank_histograms(stats1, stats2, bins=5, test=True):

    hist1 = compute_rank_histogram_buckets(stats1)
    hist2 = compute_rank_histogram_buckets(stats2)

    if test:
        label1 = "k-means(2045) + LDA(50)"
        label2 = "2D-FMC + PCA(200)"
        title = "Rank Histogram of the test set on the MSD"
    else:
        label1 = "k-means(2045) + LDA(200)"
        label2 = "2D-FMC + PCA(200)"
        title = "Rank Histogram of the train set"

    fig = plt.figure()
    ax = fig.gca()
    width = 0.45
    ax.bar(np.arange(5)-width/2, hist1, width=width, color='b', 
        label=label1, align="center")
    ax.bar(np.arange(5)+width/2, hist2, width=width, color='g', 
        label=label2, align="center")

    # Plot histogram as PDF
    plt.title(title)
    plt.xlabel("Ranks")
    plt.ylabel("Normalized Count")
    #plt.xticks(xrange(0,5), ("1-10", "11-25", "26-50", "51-100", "101+"))
    plt.xticks(xrange(0,5), ("1", "2", "3-5", "6-10", "11+"))
    plt.legend(loc="upper left")
    plt.show()

def plot_precision_at_k_histograms(stats1, stats2, K=[1,3,5,10], test=True):
    P1 = [average_precision_at_k(stats1, k) for k in K]
    P2 = [average_precision_at_k(stats2, k) for k in K]

    if test:
        label1 = "k-means(2045) + LDA(50)"
        label2 = "2D-FMC + PCA(200)"
        title = "Precision @ k of the test set on the MSD"
    else:
        label1 = "k-means(2045) + LDA(200)"
        label2 = "2D-FMC + PCA(200)"
        title = "Precision @ k of the train set"

    fig = plt.figure()
    ax = fig.gca()
    width = 0.45
    ax.bar(np.arange(len(K))-width/2, P1, width=width, color='0.75', 
        label=label1, align="center")
    ax.bar(np.arange(len(K))+width/2, P2, width=width, color='0.9', 
        label=label2, align="center", hatch='//')

    # Plot histogram as PDF
    #plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Precision @ k")
    plt.xticks(xrange(0,len(K)), ("1", "3", "5", "10"))
    ylabels = np.arange(0,3.,0.5)*10
    plt.yticks(np.arange(0,3.,0.5)*.1, ylabels.astype(int))
    plt.legend(loc="upper right")
    plt.show()


def process(statsfile, k, optfile=None):
    stats = utils.load_pickle(statsfile)
    track_ar = average_rank_per_track(stats)
    clique_ar = average_rank_per_clique(stats)
    ma_p = mean_average_precision(stats)
    #k_p = average_precision(stats, k, ver=True)
    k_p = average_precision_at_k(stats, k)

    # Set up logger
    logger = utils.configure_logger()

    # print results
    logger.info("Number of queries: %d" % len(stats))
    logger.info("Average Rank per Track: %.3f" % track_ar)
    logger.info("Average Rank per Clique: %.3f" % clique_ar)
    logger.info("Mean Average Precision: %.2f %%" % (ma_p * 100))
    logger.info("Precision at %d: %.2f %%" % (k, k_p * 100))
    
    if optfile is not None:
        stats2 = utils.load_pickle(optfile)
        #plot_rank_histograms(stats, stats2, test=False) 
        plot_precision_at_k_histograms(stats, stats2, K=[1,3,5,10], test=False)
    else:
        plot_rank_histogram(stats)

def main():
    # Args parser
    parser = argparse.ArgumentParser(description=
                "Analyzes the stats of a stats pickle file",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("statsfile", action="store",
                        help="stats file")
    parser.add_argument("-k", action="store", dest="k", default=10, type=int,
                        help="Compute Precision at k")
    parser.add_argument("-s", action="store", dest="optfile", default=None,
                        help="Optional stats file to make compare with")
    args = parser.parse_args()

    # Process
    process(args.statsfile, k=args.k, optfile=args.optfile)



if __name__ == "__main__":
    main()