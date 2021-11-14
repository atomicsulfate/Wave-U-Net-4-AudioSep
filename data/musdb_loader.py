# Reuse Wave-UNet musdb loading functions.
from musdb.audio_classes import MultiTrack


# Wave-UNet musdb loading functions sort tracks alphabetically, but
# MultiTrack class does not have a comparison function defined.
setattr(MultiTrack,"__lt__", lambda self,other: str(self) < str(other))

from waveUNet.data.musdb import *
