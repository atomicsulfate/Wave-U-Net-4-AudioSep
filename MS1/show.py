import sys, os
# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import numpy as np
import torch
from enum import Flag, auto
import matplotlib.pyplot as plt
from data.audio_prepro import stft
from data.viz import plot_wave, plot_spectrogram, plot_spectrum
from data.dataset import SeparationDataset
from data.musdb_loader import get_musdb_folds



class Plot(Flag):
    WAVE = auto()
    SPECTRUM = auto()
    SPECTROGRAM = auto()

def _plot(plot, y, sample_rate, axes, ft=None, title_suffix=None):
    '''
    Call the plotting function corresponding to the given plot type.
    :returns: The fourier transform of source if it was computed or given as argument.
    '''
    suffix = f" ({title_suffix})" if title_suffix is not None else ""
    if (isinstance(y,torch.Tensor)):
        y = y.numpy().squeeze(axis=0)
    if (plot & Plot.WAVE):
        plot_wave(y, sample_rate, axes, f"Waveform{suffix}")
    if (plot & (Plot.SPECTROGRAM | Plot.SPECTRUM)):
        if ft is None:
            ft = stft(y)
        if (plot & Plot.SPECTROGRAM):
            plot_spectrogram(ft, sample_rate, axes, f"Spectrogram{suffix}")
        if (plot & Plot.SPECTRUM):
            plot_spectrum(ft, axes, f"Spectrum{suffix}")
    return ft

def _show_signal(source, sample_rate, axes, plots, name=None):
    j = 0
    ft = None
    for plot in Plot:
        if (plots & plot):
            ax = axes[j]
            j += 1
            ft = _plot(plot, source, sample_rate, ax, ft, name)

def show(x, outfile=None, names=None, y=None, plots=Plot.SPECTROGRAM | Plot.SPECTRUM | Plot.WAVE, sample_rate=22050, dpi=300):
    '''
    Shows multiple plots describing each of the input audio signals.
    :param x: array of audio signals to show.
    :type x: list[torch.Tensor or np.array], each element with shape (1,nsamples) (mono)
    :param outfile: If 'None', plots will be shown as a Figure in the GUI. Otherwise, it's assumed to be the file path
        where the plots will be saved as an image.
    :type outfile: str
    :param names: Optional list of names assigned to the audio signals in x (used for plot titles). By default, names
        will be 'Audio {i}' with {i} being the index of the audio signal in x.
    :param y: If not 'None', it must be a list of the same length as x, with elements being dictionaries whose keys are
    strings, and values are pairs (w,sr) (same meaning as x's elements). In this case, every element in x is assumed to
    be a mixture of multiple audio sources, and its corresponding dictionary in y has the separated sources (either
    estimations or targets).
    :param plots: Flags indicating the plots to show for each audio source.
    :type plots: Plot
    :param sample_rate: Sample rate for all audio signals (default 22050)
    :type sample_rate: int
    :param dpi: Resolution used to draw matplotlib's Figure (default 100)
    :type dpi: int
    '''
    num_plots = 0
    for plot in Plot:
        if (plots & plot):
            num_plots += 1

    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    num_signals = len(x)
    if (y is not None and len(y) != num_signals):
        raise ValueError("y must have same length as x")
    if (names is not None and len(names) != num_signals):
        raise ValueError("names must have same length as x")
    subfigs = np.squeeze(plt.figure(constrained_layout=True, dpi=dpi).subfigures(nrows=num_signals, ncols=1, squeeze=False), axis=1)

    for signal_idx, subfig in enumerate(subfigs):
        signal = x[signal_idx]
        subfig_title = f'Audio {signal_idx}' if names is None else names[signal_idx]
        subfig.suptitle(subfig_title)
        num_sources = len(y[signal_idx]) if y is not None else 0
        axes = subfig.subplots(nrows=num_sources+1, ncols=num_plots, squeeze=False, sharex=False, sharey=False)
        _show_signal(signal, sample_rate, axes[0],plots, None if y is None else "mix")
        if (y is None):
            continue
        for src_idx,src in enumerate(y[signal_idx].items()):
            src_name = src[0]
            src_signal = src[1]
            _show_signal(src_signal, sample_rate, axes[src_idx+1], plots, src_name)

    if (outfile is None):
        plt.show()
    else:
        plt.savefig(outfile)


if __name__ == '__main__':
    import librosa

    # mix = librosa.load(os.path.join(root_dir, "data/musdb/test/Al James - Schoolboy Facination.stem_mix.wav"), mono=True, sr=22050)[0]
    # vocals = librosa.load(os.path.join(root_dir,"data/musdb/test/Al James - Schoolboy Facination.stem_vocals.wav"), mono=True, sr=22050)[0]
    # acc = librosa.load(os.path.join(root_dir,"data/musdb/test/Al James - Schoolboy Facination.stem_accompaniment.wav"), mono=True, sr=22050)[0]
    # show([mix], outfile=None, names=["Al James - Schoolboy Facination"], y=[{"vocals": vocals, "drums": acc}])
    #show([mix],outfile="samples.png", names=["Al James - Schoolboy Facination"], y=[{"vocals": vocals, "drums": src2}])

    data = get_musdb_folds(os.path.join(root_dir,'data/musdb'), version=None)
    test = SeparationDataset(data['test'])
    mix, targets = test[0]
    show([mix], outfile=None, y=[targets])
