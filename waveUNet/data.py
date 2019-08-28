import h5py
import musdb
import soundfile
import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset

from utils import load

class PreprocessedAudioDataset(Dataset):
    def __init__(self):
        pass

    def set_sampling_positions(self):
        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_path, "r") as f:
            lengths = [f[str(song_idx)].attrs["target_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

    def get_sample(self, index):
        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_path, 'r', driver=driver)

        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1]

        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        target_length = self.hdf_dataset[str(audio_idx)].attrs["target_length"]

        # Determine position where to start targets
        if self.random_hops:
            start_target_pos_res = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos_res = index * self.shapes["output_frames"]
        start_target_pos = start_target_pos_res * self.output_rate

        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        # READ TARGETS
        # Check if we have to pad targets at the end of a song
        end_target_pos_res = start_target_pos_res + self.shapes["output_frames"]
        if end_target_pos_res > target_length:
            pad_target_back = end_target_pos_res - target_length
            end_target_pos_res = target_length
        else:
            pad_target_back = 0

        targets = self.hdf_dataset[str(audio_idx)]["targets"][:, start_target_pos_res:end_target_pos_res].astype(
            np.float32)
        if pad_target_back > 0:
            targets = np.pad(targets, [(0, 0), (0, pad_target_back)], mode="constant", constant_values=0.0)

        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio = self.audio_transform(audio, targets)

        return audio, targets

def getMUSDB(database_path):
    mus = musdb.DB(root_dir=database_path, is_wav=False)

    subsets = list()

    for subset in ["train", "test"]:
        tracks = mus.load_mus_tracks(subset)
        samples = list()

        # Go through tracks
        for track in tracks:
            # Skip track if mixture is already written, assuming this track is done already
            track_path = track.path[:-4]
            mix_path = track_path + "_mix.wav"
            acc_path = track_path + "_accompaniment.wav"
            if os.path.exists(mix_path):
                print("WARNING: Skipping track " + mix_path + " since it exists already")

                # Add paths and then skip
                paths = {"mix" : mix_path, "accompaniment" : acc_path}
                paths.update({key : track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})

                samples.append(paths)

                continue

            rate = track.rate

            # Go through each instrument
            paths = dict()
            stem_audio = dict()
            for stem in ["bass", "drums", "other", "vocals"]:
                path = track_path + "_" + stem + ".wav"
                audio = track.targets[stem].audio
                soundfile.write(path, audio, rate, "PCM_16")
                stem_audio[stem] = audio
                paths[stem] = path

            # Add other instruments to form accompaniment
            acc_audio = np.clip(sum([stem_audio[key] for key in list(stem_audio.keys()) if key != "vocals"]), -1.0, 1.0)
            soundfile.write(acc_path, acc_audio, rate, "PCM_16")
            paths["accompaniment"] = acc_path

            # Create mixture
            mix_audio = track.audio
            soundfile.write(mix_path, mix_audio, rate, "PCM_16")
            paths["mix"] = mix_path

            diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
            print("Maximum absolute deviation from source additivity constraint: " + str(np.max(diff_signal)))# Check if acc+vocals=mix
            print("Mean absolute deviation from source additivity constraint:    " + str(np.mean(diff_signal)))

            samples.append(paths)

        subsets.append(samples)

    return subsets

def get_musdb_folds(root_path):
    dataset = getMUSDB(root_path)
    train_val_list = dataset[0]
    test_list = dataset[1]

    train_list = np.random.choice(train_val_list, 75, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]
    return {"train" : train_list, "val" : val_list, "test" : test_list}

class SeparationDataset(PreprocessedAudioDataset):
    def __init__(self, dataset, partition, instruments, sr, channels, shapes, random_hops, audio_transform=None, in_memory=False):
        '''

        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None
        self.hdf_path = partition + ".hdf5"

        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory

        self.output_rate = 1 #  1 target output per hop

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_path):
            # Create HDF file
            with h5py.File(self.hdf_path, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = channels
                f.attrs["instruments"] = instruments

                for idx, example in enumerate(dataset[partition]):
                    # Load mix
                    mix_audio, _ = load(example["mix"], sr=self.sr, mono=(self.channels == 1))

                    source_audios = []
                    for source in instruments:
                        # In this case, read in audio and convert to target sampling rate
                        source_audio, _ = load(example[source], sr=self.sr, mono=(self.channels == 1))
                        source_audios.append(source_audio)
                    source_audios = np.concatenate(source_audios, axis=0)
                    assert(source_audios.shape[1] == mix_audio.shape[1])

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
                    grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
                    grp.attrs["length"] = mix_audio.shape[1]
                    grp.attrs["target_length"] = source_audios.shape[1]

                    print("Added audio file to dataset")

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_path, "r") as f:
            if f.attrs["sr"] != sr or \
                    f.attrs["channels"] != channels or \
                    not all(f.attrs["instruments"] == instruments):
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate and channel are not as expected. Did you load an out-dated HDF file?")

        self.set_sampling_positions()

    def __getitem__(self, index):
        return self.get_sample(index)

    def __len__(self):
        return self.length