import matplotlib.pyplot as plt
import torch
from IPython.display import Audio, display
import data.audio_prepro
import librosa
import librosa.display
import numpy as np
from matplotlib.ticker import MaxNLocator
from librosa.display import TimeFormatter
from librosa.util import frame
from librosa import core
from  matplotlib import colors

def plot_wave(waveform, sample_rate, axes, title="Waveform"):
    librosa.display.waveplot(waveform, sr=sample_rate, alpha=0.8, ax=axes, color="xkcd:indigo blue")
    axes.grid()
    axes.set_title(title)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Magnitude")

def plot_wave_confidence():
    pass

def plot_spectrogram(stft, sample_rate, axes, title="Spectrogram"):
    n_fft = (stft.shape[0]-1)*2
    hop_length = int(n_fft / data.audio_prepro.DEF_HOP_LENGTH_DIV)
    D = np.abs(stft)
    DB = librosa.amplitude_to_db(D, ref=np.max)
    m = librosa.display.specshow(DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log', ax=axes);
    plt.colorbar(m, format='%+2.0f dB', ax=axes)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Frequency (Hz)")
    axes.set_title(title)

def plot_spectrum(stft, axes, title="Spectrum"):
    D = np.abs(stft)
    axes.plot(D, c="xkcd:indigo blue");
    axes.grid()
    axes.set_title(title)
    axes.set_xlabel("Frequency (Hz)")
    axes.set_ylabel("Amplitude")

def plot_wave_confidence(wave, confidence, sample_rate, axes, title="Waveform", cmap_name='plasma', fig=None):
    '''
    Plots an audio waveform colored according to the given confidence for each window interval.
    :param wave: The input audio
    :param confidence: An np.array with the confidence value for each time window.
    :param sample_rate: Sampling rate for all audio signals.
    :param axes: Axes used for the plot.
    :param title: Plot title.
    :param cmap_name: Name of the color map to represent the confidence.
    :param fig: Figure that will contain the plot. If given, the confidence colorbar legend will be drawn as part
                of the figure.
    '''
    if wave.ndim == 1:
        wave = wave[np.newaxis, :]

    sr = sample_rate
    num_windows = len(confidence)
    hop_length = 25 # Num audio samples per point in the graph. Divides typical sample rates like 44100 and 22050.
    num_samples_win = wave.shape[1] // num_windows
    num_points_win = num_samples_win // hop_length

    # axis formatting
    axes.xaxis.set_major_formatter(TimeFormatter(unit=None, lag=False))
    axes.xaxis.set_major_locator(MaxNLocator(nbins='auto', prune=None, steps=[1, 1.5, 5, 6, 10], min_n_ticks=10))
    axes.tick_params(labelsize=14)
    axes.set_title(title, fontsize=20)
    axes.set_xlabel('Time', fontsize=18)
    axes.set_ylabel('Amplitude', fontsize=18)

    y = np.abs(frame(wave, frame_length=hop_length, hop_length=hop_length)).max(axis=1)
    y_top = y[0]
    y_bottom = -y[-1]

    x = core.times_like(y_top, sr=sr, hop_length=hop_length)
    axes.set_xlim([x.min(), x.max()])

    norm = colors.Normalize(vmin=confidence.min(), vmax=confidence.max())
    cmap = plt.cm.get_cmap(cmap_name)

    for i in range(num_windows):
        axes.fill_between(x[i * num_points_win:(i + 1) * num_points_win],
                         y_bottom[i * num_points_win:(i + 1) * num_points_win],
                         y_top[i * num_points_win:(i + 1) * num_points_win], color=cmap(norm(confidence[i])))

    if (fig is not None):
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Confidence')
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.yaxis.get_label().set_fontsize(20)

def plot_prediction_confidence(input_mixture, prediction, confidence, sr):
    '''
    Plots the waveforms corresponding to the input mixture and predicted sources colored to indicate
    the confidence in the prediction for each window interval.
    :param input_mixture: The input mixture audio
    :param prediction: Predicted audio sources in a dictionary, where the keys are the source names and the values
           the corresponding audio time series.
    :param confidence: An np.array with the confidence value for each time window.
    :param sr: Sampling rate for all audio signals.
    :returns: The created Figure.
    '''
    cmap_name = "plasma"
    num_plots = len(prediction)+1
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10*num_plots, 15))

    plot_wave_confidence(input_mixture.detach().cpu().numpy(), confidence, sr, axes[0], "Input mixture", cmap_name)
    for i, (source_name, source_audio) in enumerate(prediction.items()):
        plot_wave_confidence(source_audio.squeeze(), confidence, sr, axes[i+1], f'Est. {source_name}', cmap_name)

    norm = colors.Normalize(vmin=confidence.min(), vmax=confidence.max())
    cmap = plt.cm.get_cmap(cmap_name)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85, hspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical', label='Confidence')
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.get_label().set_fontsize(20)
    return fig

#################
# Misc. functions
#################

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    if not isinstance(waveform, np.ndarray):
        waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    if not isinstance(waveform, np.ndarray):
        waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


def play_audio(waveform, sample_rate):
    if not isinstance(waveform, np.ndarray):
        waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def plot_mel_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)


def plot_pitch(waveform, sample_rate, pitch):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ln2 = axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    axis2.legend(loc=0)
    plt.show(block=False)


def plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Kaldi Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")
    axis.set_ylim((-1.3, 1.3))

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, nfcc.shape[1])
    ln2 = axis2.plot(
        time_axis, nfcc[0], linewidth=2, label="NFCC", color="blue", linestyle="--"
    )

    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    axis.legend(lns, labels, loc=0)
    plt.show(block=False)

DEFAULT_OFFSET = 201
SWEEP_MAX_SAMPLE_RATE = 48000
DEFAULT_LOWPASS_FILTER_WIDTH = 6
DEFAULT_ROLLOFF = 0.99
DEFAULT_RESAMPLING_METHOD = "sinc_interpolation"


def plot_sweep(
    waveform,
    sample_rate,
    title,
    max_sweep_rate=SWEEP_MAX_SAMPLE_RATE,
    offset=DEFAULT_OFFSET,
):
    x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
    y_ticks = [1000, 5000, 10000, 20000, sample_rate // 2]

    time, freq = data.audio_prepro._get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
    freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
    freq_y = [f for f in freq if f >= 1000 and f in y_ticks and f <= sample_rate // 2]

    figure, axis = plt.subplots(1, 1)
    axis.specgram(waveform[0].numpy(), Fs=sample_rate)
    plt.xticks(time, freq_x)
    plt.yticks(freq_y, freq_y)
    axis.set_xlabel("Original Signal Frequency (Hz, log scale)")
    axis.set_ylabel("Waveform Frequency (Hz)")
    axis.xaxis.grid(True, alpha=0.67)
    axis.yaxis.grid(True, alpha=0.67)
    figure.suptitle(f"{title} (sample rate: {sample_rate} Hz)")
    plt.show(block=True)

    

