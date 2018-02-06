import musdb
import numpy as np
import functools
from scipy.signal import stft, istft


def IRM(track, alpha=2):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal ratio mask.
    this is the ratio of spectrograms, where alpha is the exponent to take for
    spectrograms. usual values are 1 (magnitude) and 2 (power)"""

    # STFT parameters
    nfft = 2048

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0]  # remember number of samples for future use
    X = stft(track.audio.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT to the power alpha
        P[name] = np.abs(stft(source.audio.T, nperseg=nfft)[-1])**alpha
        model += P[name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        # compute soft mask as the ratio between source spectrogram and total
        Mask = np.divide(np.abs(P[name]), model)

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment'] = accompaniment_source

    return estimates


# initiate dsdtools
mus = musdb.DB()

alpha = 2

mus.run(
    functools.partial(IRM, alpha=alpha),
    estimates_dir='IRM_alpha=%d' % alpha,
    subsets='test',
    parallel=True,
    cpus=4
)
