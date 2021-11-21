import musdb_loader
import torch
import pandas as pd
import glob
import os
from utils import load, write_wav
import random
import numpy as np
from pathlib import Path

def get_speaker_speeches(path_to_train_clean):
    speaker_speeches = []
    people = glob.glob(os.path.join(path_to_train_clean, "*", ""))
    for person in people:
        sub_dirs = glob.glob(os.path.join(person, "*", ""))
        current_speakers_speech = []
        for sub_dir in sub_dirs:
            current_speakers_speech.extend(glob.glob(os.path.join(sub_dir, "*.flac")))
        speaker_speeches.append(current_speakers_speech)
    return speaker_speeches

def get_speech(speaker_data, song_length):
    speech_length = 0
    speech_tensor = torch.zeros((song_length))
    for speech_segment in speaker_data:
        audio_tensor = load(speech_segment, mode="pytorch")[0]
        curr_total = speech_length + audio_tensor.shape[1]
        if (curr_total < song_length):
            speech_tensor[speech_length:curr_total] = audio_tensor
            speech_length = curr_total
        else:
            # rest of the tensor will be zeros
            break
    return speech_tensor[None, :]

def get_new_song(speaker_data, musdb_data):
    indices = np.random.randint(0, high=len(musdb_data)-1, size=4)
    current_combination = {}
    song_length = float("inf")
    for i, instrument in enumerate(list(musdb_data)):
        instrument_path = musdb_data[instrument].loc[indices[i]]
        logging.info(f"Loading instrument: {instrument_path}")
        audio_tensor = load(instrument_path, mode="pytorch")[0]
        song_length = min(song_length, audio_tensor.shape[1])
        current_combination[instrument] = audio_tensor

    current_combination['accompaniment'] = torch.zeros((1, song_length))
    for key, value in current_combination.items():
        current_combination[key] = torch.squeeze(value)[: song_length][None, :]
        current_combination['accompaniment'] += current_combination[key]
    current_combination['vocals'] = get_speech(speaker_data, song_length)
    current_combination['mix'] = current_combination['vocals'] + current_combination['accompaniment']
    return current_combination

def save_song(combination, track_path):
    for stem in combination.keys():
        path = track_path + "_" + stem + ".wav"
        audio = combination[stem]
        write_wav(path, audio.T, 22500)

def create_extended_musdb(path_to_train_clean, musdb_data, save_path):
    speaker_speeches = get_speaker_speeches(path_to_train_clean)
    test_ratio = int(0.8*len(speaker_speeches))
    for i, speaker in enumerate(speaker_speeches):
        combination = get_new_song(speaker, musdb_data)
        if (i < test_ratio):
            track_path = os.path.join(save_path, "train", str(i))
        else:
            track_path = os.path.join(save_path, "test", str(i))
        save_song(combination, track_path)
    return

def get_musdb_extended(path):
    subsets = list()

    for subset in ["train", "test"]:
        tracks = glob.glob(os.path.join(path, subset, "*_mix.wav"))
        samples = list()
        # Go through tracks
        for track in sorted(tracks):
            track_name = Path(track).name
            # Skip track if mixture is already written, assuming this track is done already
            track_path = os.path.join(path, subset, track_name.split('_')[0])
            mix_path = track_path + "_mix.wav"
            acc_path = track_path + "_accompaniment.wav"
            if os.path.exists(mix_path):
                logging.debug("WARNING: Skipping track " + mix_path + " since it exists already")

                # Add paths and then skip
                paths = {"mix" : mix_path, "accompaniment" : acc_path}
                paths.update({key : track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})
                samples.append(paths)
                continue
            else:
                raise ValueError('Generate the extended musdb dataset first')
        subsets.append(samples)
    return subsets

def get_extended_musdb_folds(root_path, path_to_train_clean=None, musdb_path=None):
    try:
        dataset = get_musdb_extended(root_path)
    except ValueError:
        logger.info("Extended dataset not created yet, generating it now")
        if not musdb_path or not path_to_train_clean:
            raise ValueError('Missing extra paths to generate extended db')
        data = musdb_loader.get_musdb_folds(musdb_path, version=None)
        musdb_data = pd.DataFrame(list(data['train'])).drop('mix', 1)
        musdb_data = musdb_data.drop('vocals', 1)
        musdb_data = musdb_data.drop('accompaniment', 1)
        create_extended_musdb(path_to_train_clean, musdb_data, root_path)
        dataset = get_musdb_extended(path)

    train_val_list = dataset[0]
    test_list = dataset[1]

    np.random.seed(1337) # Ensure that partitioning is always the same on each run
    val_ratio = 0.25
    train_size = round(len(train_val_list)*(1-val_ratio))
    train_list = np.random.choice(train_val_list, train_size, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]
    # print("First training song: " + str(train_list[0])) # To debug whether partitioning is deterministic
    return {"train" : train_list, "val" : val_list, "test" : test_list}
