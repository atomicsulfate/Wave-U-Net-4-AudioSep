# Reuse Wave-UNet musdb loading functions.
from musdb.audio_classes import MultiTrack
import musdb

# Wave-UNet musdb loading functions sort tracks alphabetically, but
# MultiTrack class does not have a comparison function defined.
setattr(MultiTrack,"__lt__", lambda self,other: str(self) < str(other))

from waveUNet.data.musdb import *

def setup_hq_musdb(root_path):
    '''
    Creates a directory for each track, leaving inside a wav file for each target. This dataset structure is expected
    by musdb, museval and sigsep-mus-oracle.
    '''
    db = musdb.DB(root=root_path, is_wav=False)
    for track in db:
        targets = {target_name: target.audio for target_name, target in track.targets.items()}
        del targets['linear_mixture']
        targets['mixture'] = track.audio
        db.save_estimates(targets, track, root_path)
