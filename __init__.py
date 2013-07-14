'''
Created on Feb 15, 2012

@author: humphreyadmin
'''


import getpass
import os

if getpass.getuser() == 'ejhumphrey':
    _base_path = ''
elif getpass.getuser() == 'humphreyadmin':
    _base_path = '/Users/humphreyadmin/Desktop/musicsummary'
elif getpass.getuser() == 'uri':
    _base_path = '/Users/uri/NYU/ResearchCode/MusicSummary'
elif getpass.getuser() == 'uriadmin':
    _base_path = '/Users/uriadmin/NYU/ResearchCode/MusicSummary'
    _base_path = '/Volumes/Audio/MusicSummary'

class FileStrings():
    base_dir = _base_path

    # Extensions
    brute_out_base = '-bout.txt'
    eval_out_base = '-eout.txt'
    beat_base = '-beats.txt'
    seq_base = '-seq.txt'
    sum_base = '-sum.txt'
    tonnetz_dict_base = "-tz-dict.txt"
    chroma_dict_base = "-ch-dict.txt"
    tonnetz_sum_base = "-tz-sum.txt"
    chroma_sum_base = "-ch-sum.txt"
    audio_base = '.mp3'
    dict_base = 'dict.txt'
    codebook = 'codebook.npy'
    features = "raw_features.npy"
    tmp_audio_base = '.wav'
    sample_file = 'sampled_tracknames.pkl'
    # Binaries
    brute_bin = './brute_force'
    heuristic_bin = './heuristic'
    random_bin = './random'
    eval_bin = './evaluation'

    # Options
    brute_opt = 'brute'
    rand_opt = 'rand'
    heur_opt = 'heur'

    # Data base folders
    #dataset_folder = 'covers'   # For Cover Song Data Set
    #dataset_folder = 'dataset'  # For Mazurka Data Set
    dataset_folder = 'dataset2'  # For Mazurka Data Set
    dataset_folder = 'mazurkas_v2'  # For Mazurka Data Set
    dataset_folder = 'mazurkas_v2-k50'
    dataset_folder = 'mazurkas_v2-heur'
    dataset_folder = 'mazurkas_v2-rand'
    dataset_folder = 'mazurkas-all-'
    dataset_folder = 'soundcloud'
    audio_folder = 'audio'
    beats_auto_folder = 'beats_auto'
    beats_gt_folder = 'beats_gt'
    out_folder = 'out'
    gt_folder = 'gt'
    auto_folder = 'auto'
    fold_pattern = 'fold_%02d'
    dataset_path = os.path.join( base_dir, dataset_folder )
    audio_path = os.path.join( dataset_path, audio_folder )
    beats_auto_path = os.path.join( dataset_path, beats_auto_folder )
    beats_gt_path = os.path.join( dataset_path, beats_gt_folder )
    sample_file_path = os.path.join( dataset_path, sample_file )

    fold_file = os.path.join(base_dir, dataset_path, 'folds.txt')

    # Features
    tonnetz = 'tonnetz'
    difftonnetz = 'tonnetz_diff'
    chroma = 'chroma'

    def __init__(self):
        pass

    def format_audio(self, track):
        if track.count(self.audio_folder)==0:
            track = os.path.join(self.base_dir, self.dataset_path, self.audio_folder, track)
        return track

    def audio_to_autobeats(self, track):
        track = self.format_audio(track)
        temp = track.replace(self.audio_folder,self.beats_auto_folder)
        return temp.replace(self.audio_base,self.beat_base)

    def audio_to_gtbeats(self, track):
        track = self.format_audio(track)
        temp = track.replace(self.audio_folder,self.beats_gt_folder)
        return temp.replace(self.audio_base,self.beat_base)




