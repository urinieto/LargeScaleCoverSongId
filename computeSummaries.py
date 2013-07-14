#!/usr/bin/env python

"""computeSummaries.py: Computes the summaries from the pickle files
from the MSD."""

__author__      = "Uri Nieto"
__date__        = "15/11/12"

import time
import argparse
import os
import glob
import pylab as plt
import numpy as np
import utils
import cPickle
import subprocess
from __init__ import FileStrings
from scipy.spatial import distance


COMPRESS_CMD = "xz"
EXT_COMP = ".xz"
EXT_TMP = ".tmp"
EXT_SUM = ".sum"
SHS_TEST_FILE = "SHS/test.pk"
SHS_TRAIN_FILE = "SHS/train.pk"
OUT_SHS_SUM = "SHS/summaries/"


def compute_dist( feat1, feat2 ):
    """Computes the Euclidean distance from feat1 and feat2."""
    #return distance.euclidean( feat1, feat2 )
    return np.linalg.norm( feat1 - feat2 )  # This is twice as fast!


def compute_dict( X, out_file, feat_name="Tonnetz" ):
    """Computes the dictionary from X and writes it to out_file."""
    f = open(out_file, "w")
    out_str = "# Generated from %s\n"%(feat_name)
    out_str += "# Classes: %d\n"%(X.shape[0])
    for i, x1 in enumerate(X):
        for j in range(i+1,X.shape[0]):
            x2 = X[j]
            d = compute_dist( x1, x2 )
            out_str += "%d,%d,%f\n"%(i, j, d)
    f.write(out_str)
    f.close()
    compressed_file = out_file + EXT_COMP
    if os.path.isfile( compressed_file ):
        os.remove(compressed_file)
    cmd = "%s %s"%(COMPRESS_CMD, out_file)
    subprocess.call( cmd.split(" ") )


def compute_dicts( in_file, tonnetz_file, chroma_file ):
    """Compute the dictionary from a pickle file."""
    # Read data
    pfile = open(in_file, "r")
    try:
        data = cPickle.load(pfile)
        #if not os.path.isfile( tonnetz_file ):
        compute_dict( data["tonnetz"], tonnetz_file, feat_name="Tonnetz" )
        #if not os.path.isfile( chroma_file ):
        compute_dict( data["pitches"], chroma_file, feat_name="Chroma" )
    except Exception, e:
        print "Can't load cPickle file: ", in_file

    pfile.close()


def compute_summary( dict_file, heur_bin, ext_name, P=4, N=16 ):
    """Computes the summary using the heur_bin, given
    the dictionary of distances dict_file, saves it into the
    in_file+ext_name file, using P and N as parameters."""
    # Decompress dictionary
    dict_file += EXT_COMP
    cmd = "%s -d %s"%(COMPRESS_CMD, dict_file)
    subprocess.call( cmd.split(" ") )
    dict_file = dict_file.replace(EXT_COMP, "")

    # Read number of beats
    f = open( dict_file, "r" )
    lines = f.readlines()
    nbeats = lines[1].split(":")[1]
    nbeats = int(nbeats.split("\n")[0])
    print "nbeats: ", nbeats
    f.close()

    # Create temp file for the sequence of beats
    tmp_name = dict_file + EXT_TMP
    f = open( tmp_name, "w" )
    f.write(str(range(nbeats))[1:-1] + "\n")
    f.close()

    # Call the heuristic process
    out_name = OUT_SHS_SUM + os.path.basename(dict_file) + EXT_SUM
    cmd = "%s %s %s %s %s %s"%(heur_bin, dict_file, tmp_name, out_name, P, N)
    subprocess.call( cmd.split(" ") )

    # Remove temp file
    os.remove(tmp_name)

    # Compress data again
    cmd = "%s %s"%(COMPRESS_CMD, dict_file)
    subprocess.call( cmd.split(" ") )


def get_paths_from_pk( in_file, msd_path, ext=".pk" ):
    """Gets the file paths from a pickle file."""
    pfile = open(in_file, "r")
    data = cPickle.load(pfile)
    paths = []
    for key in data.keys():
        for track in data[key]:
            path = os.path.join(msd_path, track[2], track[3],
                                track[4], track + ext)
            paths.append( path )
    pfile.close()
    return paths


def get_shs_path( msd_path ):
    """Gets the paths for the SHS files."""
    paths = get_paths_from_pk( SHS_TEST_FILE, msd_path )
    paths += get_paths_from_pk( SHS_TRAIN_FILE, msd_path )
    return paths


def compute_dicts_from_file( in_file, heur_bin="heuristic", dict=False,
                            ext=".pk", P=4, N=16 ):
    """Computes the dictionaries from a specific pickle file with beat-sync feats."""

    # Hack for when the distances are still being computed
    try:
        with open(in_file) as f: pass
    except IOError as e:
        print 'Oh dear.'
        root_dir = in_file.split("/")[0]
        in_file = in_file[:len(root_dir)+1] + "MSD/" + in_file[len(root_dir)+1:]

    dirname = os.path.dirname( in_file )
    tonnetz_file = os.path.join(dirname,
                                os.path.basename(in_file)).replace(
                                    ext,
                                    FileStrings.tonnetz_dict_base)
    chroma_file = os.path.join(dirname,
                               os.path.basename(in_file)).replace(
                                ext,
                                FileStrings.chroma_dict_base)
    print in_file, tonnetz_file, chroma_file
    if dict:
        compute_dicts(in_file, tonnetz_file, chroma_file)
    compute_summary(tonnetz_file, heur_bin,
                    ext_name=FileStrings.tonnetz_sum_base,
                    P=P, N=N)
    compute_summary(chroma_file, heur_bin,
                    ext_name=FileStrings.chroma_sum_base,
                    P=P, N=N)


def process(msd_path=".", heur_bin="heuristic", dict=False, ext=".pk",
            P=4, N=16, shs=False):
    """Main process to compute the summaries from pickle files."""
    if shs:
        paths = get_shs_path( msd_path )
        for in_file in paths:
            compute_dicts_from_file( in_file, heur_bin, dict, ext, P, N )
    else:
        for root, dirs, files in os.walk(msd_path):
            files = glob.glob(os.path.join(root,'*'+ext))
            try:
                print files[0]
            except:
                print "OK"
            for in_file in files:
                compute_dicts_from_file( in_file, heur_bin, dict, ext, P, N )


def main():
    """Main function to compute the summaries."""
    # Args parser
    parser = argparse.ArgumentParser(description=
             "Computes the summaries from the pickle files of the MSD.")
    parser.add_argument("msd_path", action="store",
                        help="Path to the MSD with pickle files.")
    parser.add_argument("heuristic_bin", action="store",
                        help="Binary to compute the summary using the " \
                        "heuristic approach.")
    parser.add_argument("-d", action="store_true", dest="dict",
                        help="Create the dictionaries from the features "\
                        "(default: False)")
    parser.add_argument("-P", action="store", dest="nsubseqs", default=4,
                        help="Number of subsequences for each summary "\
                        "(default: P=4)")
    parser.add_argument("-N", action="store", dest="nbeats", default=16,
                        help="Number of beats per subsequence "\
                        "(default: N=16)")
    parser.add_argument("-s", action="store_true", dest="shs",
                        help="Compute the SHS summaries only "\
                        "(default: False)")
    args = parser.parse_args()

    # Run the process
    start = time.time()
    process(msd_path=args.msd_path, heur_bin=args.heuristic_bin,
            dict=args.dict, P=args.nsubseqs, N=args.nbeats, shs=args.shs)
    print "Time taken: %.3f sec"%(time.time() - start)


if __name__ == "__main__":
    main()
