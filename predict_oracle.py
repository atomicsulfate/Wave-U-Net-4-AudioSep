import os
import sys
import musdb
import museval
import models.oracles as oracles
from data.eval import evaluate_estimates, evaluate_track_estimates
from data.musdb_utils import save_estimates
import numpy as np
from tqdm import tqdm
import collections

class TrackAdapter(musdb.MultiTrack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Oracle code expects 2-d audio (nsamples,nchannels), reshape on demand if needed.
        new_sources = {}
        for source in track.sources.values():
            new_sources[source.name] = SourceAdapter(self, source.name, source.path, source.stem_id,
                                                     sample_rate=track.sample_rate)
        new_targets = collections.OrderedDict()
        for name, target in self.targets.items():
            target_sources = [new_sources[source.name] for source in target.sources]
            new_targets[name] = musdb.Target(self, target_sources, name)
        self.targets = new_targets
        self.sources = new_sources

    @property
    def audio(self):
        audio = super().audio
        return audio if (len(audio.shape) > 1) else np.array([audio, audio]).T

class SourceAdapter(musdb.Source):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def audio(self):
        audio = super().audio
        return audio if (len(audio.shape) > 1) else np.array([audio, audio]).T

if __name__ == '__main__':
    db_path = sys.argv[1]
    estimates_path = sys.argv[2]
    oracle = oracles.get_id(sys.argv[3])

    db = musdb.DB(root=db_path, subsets="test", is_wav=True)

    for track in tqdm(db):
        new_track = TrackAdapter(track.path, track.name, track.artist, track.title, None, track.targets,
                                 track.sample_rate, is_wav=track.is_wav, stem_id=track.stem_id, subset=track.subset)
        references = {name: target.audio for name, target in new_track.targets.items()}
        test_path = os.path.join(estimates_path,"test")

        if (os.path.exists(os.path.join(test_path, new_track.name))):
            museval._load_track_estimates(new_track, estimates_path, estimates_path)
        else:
            estimates = oracles.predict_track(new_track, oracle)
            save_estimates(new_track.name, test_path, estimates)
            evaluate_track_estimates(new_track.name, references, estimates, estimates_path)

