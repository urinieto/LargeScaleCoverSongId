#!/usr/bin/env python

"""beatsyncPithces.py: Create the data set for summaries from the MSD."""
__author__      = "Uri Nieto"
__date__        = "24/10/12"

import tables
import time
import argparse
import os
import glob
import hdf5_getters as h5t
import pylab as plt
import numpy as np
import utils
import cPickle
import dan_tools

MSD_BASE_DIR = "MSD"
MSD_SUM_BASE_DIR = "MSD_SUM"

def count_all_files(basedir, ext=".h5"):
    cnt = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        cnt += len(files)
    return cnt


def get_all_titles(basedir, ext=".h5"):
    titles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            h5 = h5t.open_h5_file_read(f)
            titles.append( h5t.get_title(h5) )
            h5.close()
    return titles


def dan_extract_data(filename):
    data = dict()
    data["pitches"] = dan_tools.msd_beatchroma(filename).T
    f = h5t.open_h5_file_read( filename )
    data["tonnetz"] = utils.chroma_to_tonnetz( data["pitches"] )
    data["track_id"] = h5t.get_track_id( f )
    data["artist_name"] = h5t.get_artist_name( f )
    data["title"] = h5t.get_title( f )
    f.close()


def extract_data(filename):
    # Open hdf5 file
    f = h5t.open_h5_file_read( filename )

    # Get beat synchronous harmonic features
    dur = h5t.get_duration( f )
    seg_starts = h5t.get_segments_start( f )
    beats = h5t.get_beats_start( f )
    pitches = h5t.get_segments_pitches( f )

    #print "beats: ", beats.shape, "segments: ", seg_starts.shape, "pitches:", \
    #        pitches.shape

    data = utils.beatsync_segments(seg_starts, beats, pitches, dur,
                                   weight=False)

    # Get rest of data
    data["tonnetz"] = utils.chroma_to_tonnetz( data["pitches"] )
    data["track_id"] = h5t.get_track_id( f )
    data["artist_name"] = h5t.get_artist_name( f )
    data["title"] = h5t.get_title( f )

    #utils.plot_features( pitches )
    #utils.plot_features( beatsync["pitches"] )
    #utils.plot_features( beatsync["tonnetz"] )

    f.close()

    #print "beats: ", beats.shape, "bars: ", bars.shape, "sections: ", \
    #    sections.shape, "tatums: ", tatums.shape, artist, title, track_id, \
    #    seg_starts[-2], pitches.shape, beats.shape

    return data


def create_sum_dataset( basedir, ext=".h5", out_ext=".pk" ):
    """Creates the dataset for summaries"""
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        try:
            print files[0]
        except:
            print "OK"
        for f in files:
            dirname = os.path.dirname( f )
            dirname = dirname.replace( MSD_BASE_DIR, MSD_SUM_BASE_DIR )
            utils.create_dir( dirname )
            newfile = os.path.join(dirname,
                                   os.path.basename(f)).replace(ext, out_ext)
            pfile = open( newfile, "w" )
            data = extract_data( f )
            cPickle.dump( data, pfile )
            pfile.close()


def main():
    """Main function to extract features from the MSD."""
    # Args parser
    parser = argparse.ArgumentParser(description=
             'Computes and extracts beat-sync Tonnetz and Chroma features' \
             'from the MSD.')
    args = parser.parse_args()

    # Run the process
    start = time.time()
    create_sum_dataset( MSD_BASE_DIR )
    print "Time taken: %.3f sec"%(time.time() - start)


if __name__ == "__main__":
    main()
