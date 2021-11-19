# -------------------------------------------------------------------------------
# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
# -------------------------------------------------------------------------------
import os
import math
import time
import torch
import torchaudio
import librosa
import logging
logger = logging.getLogger(__name__)

DEF_HOP_LENGTH_DIV = 4;

def stft(waveform, n_fft = 2048):
    # If hop_length is changed, it must be changed in plot_spectrogram()!
    hop_length = int(n_fft / DEF_HOP_LENGTH_DIV)  # number of audio samples between adjacent STFT columns.
    return librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)

###################################
# Misc. functions (unused so far)
###################################
SAMPLE_WAV_SPEECH_PATH=None
SAMPLE_RIR_PATH=None
DEFAULT_OFFSET=None
DEFAULT_LOWPASS_FILTER_WIDTH=None
DEFAULT_ROLLOFF=None
DEFAULT_RESAMPLING_METHOD=None
def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend([["lowpass", f"{resample // 2}"], ["rate", f"{resample}"]])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def get_speech_sample(*, resample=None):
    return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)


def get_sample(*, resample=None):
    return _get_sample(SAMPLE_WAV_PATH, resample=resample)


def get_rir_sample(*, resample=None, processed=False):
    rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
    if not processed:
        return rir_raw, sample_rate
    rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir, sample_rate


def get_noise_sample(*, resample=None):
    return _get_sample(SAMPLE_NOISE_PATH, resample=resample)


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
    max_sweep_rate = sample_rate
    freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
    delta = 2 * math.pi * freq / sample_rate
    cummulative = torch.cumsum(delta, dim=0)
    signal = torch.sin(cummulative).unsqueeze(dim=0)
    return signal


def benchmark_resample(
    method,
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=DEFAULT_LOWPASS_FILTER_WIDTH,
    rolloff=DEFAULT_ROLLOFF,
    resampling_method=DEFAULT_RESAMPLING_METHOD,
    beta=None,
    iters=5,
):
    if method == "functional":
        begin = time.time()
        for _ in range(iters):
            F.resample(
                waveform,
                sample_rate,
                resample_rate,
                lowpass_filter_width=lowpass_filter_width,
                rolloff=rolloff,
                resampling_method=resampling_method,
            )
        elapsed = time.time() - begin
        return elapsed / iters
    elif method == "transforms":
        resampler = T.Resample(
            sample_rate,
            resample_rate,
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            resampling_method=resampling_method,
            dtype=waveform.dtype,
        )
        begin = time.time()
        for _ in range(iters):
            resampler(waveform)
        elapsed = time.time() - begin
        return elapsed / iters


def _get_log_freq(sample_rate, max_sweep_rate, offset):
    """Get freqs evenly spaced out in log-scale, between [0, max_sweep_rate // 2]

  offset is used to avoid negative infinity `log(offset + x)`.

  """
    half = sample_rate // 2
    start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
    return (
        torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset
    )


def _get_inverse_log_freq(freq, sample_rate, offset):
    """Find the time where the given frequency is given by _get_log_freq"""
    half = sample_rate // 2
    return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))


def _get_freq_ticks(sample_rate, offset, f_max):
    # Given the original sample rate used for generating the sweep,
    # find the x-axis value where the log-scale major frequency values fall in
    time, freq = [], []
    for exp in range(2, 5):
        for v in range(1, 10):
            f = v * 10 ** exp
            if f < sample_rate // 2:
                t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
                time.append(t)
                freq.append(f)
    t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
    time.append(t_max)
    freq.append(f_max)
    return time, freq


def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")

def write_wav(path, audio, sr):
    logger.debug(f"Saving audio file in location: {path} with sample_rate {sr}")
    torchaudio.save(path, torch.from_numpy(audio.T).to(torch.float32), sr)
