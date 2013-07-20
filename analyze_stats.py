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

def average_precision(stats, q):
    try:
        nrel = len(stats[q]) # Number of relevant docs
    except:
        return np.nan
    ap = []
    for k in stats[q]:
        pk = precision_at_k(stats[q], k)
        ap.append(pk)
    return np.sum(ap) / float(nrel)

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

def stat_differences(s1, s2):
    tr1 = get_top_ranked(s1)
    tr2 = get_top_ranked(s2)
    ar1 = get_average_rank(s1)
    ar2 = get_average_rank(s2)

    fig, ax = plt.subplots(3)
    max = 12000
    #print tr1[71], tr2[71]
    #print ar1[1977], ar2[1977] # Extreme case!
    p1, = ax[0].plot(tr1[:max], 's', alpha=0.6)
    p2, = ax[0].plot(tr2[:max], 's', alpha=0.6)
    ax[0].set_title("Top Ranked")
    ax[0].legend([p1, p2], ["PCA", "BasisProj"])

    p1, = ax[1].plot(ar1[:max], 's', alpha=0.6)
    p2, = ax[1].plot(ar2[:max], 's', alpha=0.6)
    ax[1].set_title("Average Rank")
    ax[1].legend([p1, p2], ["PCA", "BasisProj"])

    p1, = ax[2].plot(abs(ar1[:max] - ar2[:max]), 's', alpha=0.6)
    p2, = ax[2].plot(abs(tr1[:max] - tr2[:max]), 's', alpha=0.6)
    ax[2].set_title("Difference between PCA and BasisProj")
    ax[2].legend([p1, p2], ["Top Rank", "Average Rank"])
    plt.show()

def compute_rank_histogram(stats, bins=100):
    ranks = []
    for s in stats:
        try:
            for rank in s:
                ranks.append(rank)
        except:
            continue
    # Calculate histogram
    hist, edges = np.histogram(ranks, bins=bins)
    
    # Probability Density Function:
    hist = hist.astype(float)
    hist /= float(hist.sum())

    # Plot histogram as PDF
    plt.bar(xrange(0,bins), hist)
    plt.title("Rank Histogram")
    plt.xlabel("Ranks")
    plt.ylabel("Normalized Count")
    if len(stats) > 6000:
        # If stats from the Training set, set the ticks from 0 to 12930
        plt.xticks(xrange(0, bins, bins/5), xrange(0, 15000, 15000/5))
        plt.yticks(np.arange(0, 4)/10.)
    else:
        # Set stats for the Million Song Dataset (0 to 1000000)
        plt.xticks(xrange(0, bins, bins/5), xrange(0, 1000000, 1000000/5))
    plt.show()

def compute_rank_histogram_buckets(stats, bins=5):
    ranks = []
    for s in stats:
        try:
            for rank in s:
                ranks.append(rank)
        except:
            continue
    # Calculate histogram
    hist = np.zeros(5) #1-10, 10-25, 25-50, 50-100, 100+
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

    # Probability Density Function:
    hist = hist.astype(float)
    hist /= float(hist.sum())

    # Plot histogram as PDF
    plt.bar(xrange(0,bins), hist, align="center")
    plt.title("Rank Histogram")
    plt.xlabel("Ranks")
    plt.ylabel("Normalized Count")
    plt.xticks(xrange(0,5), ("1-10", "11-25", "26-50", "51-100", "101+"))
    plt.show()


def process(statsfile, k, optfile=None):
    stats = utils.load_pickle(statsfile)
    track_ar = average_rank_per_track(stats)
    clique_ar = average_rank_per_clique(stats)
    ma_p = mean_average_precision(stats)
    k_p = average_precision(stats, k)

    if optfile != None:
        stats2 = read_cPickle(optfile)
        stat_differences(stats, stats2)

    print "Number of queries:", len(stats)
    print "Average Rank per Track: %.3f" % track_ar
    print "Average Rank per Clique: %.3f" % clique_ar
    print "Mean Average Precision: %.2f %%" % (ma_p * 100)
    print "Precision at %d: %.2f %%" % (k, k_p * 100)
    
    #compute_rank_histogram(stats)
    compute_rank_histogram_buckets(stats)
    #N = len(stats)
    #m = mean_per_clique_count(stats, N=N)
    #plt.bar(xrange(0, N), m)
    #plt.show()

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