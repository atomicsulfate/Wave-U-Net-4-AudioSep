import sys, os
# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from enum import Flag, auto
import matplotlib.pyplot as plt
from data.audio_prepro import stft
from data.viz import plot_wave, plot_spectrogram, plot_spectrum
import numpy as np

class Plot(Flag):
    WAVE = auto()
    SPECTRUM = auto()
    SPECTROGRAM = auto()

def _plot(plot, source, axes, ft=None):
    '''
    Call the plotting function corresponding to the given plot type.
    :returns: The fourier transform of source if it was computed or given as argument.
    '''
    y,sr = source
    if (plot & Plot.WAVE):
        plot_wave(y, sr, axes)
    if (plot & (Plot.SPECTROGRAM | Plot.SPECTRUM)):
        if ft is None:
            ft = stft(y)
        if (plot & Plot.SPECTROGRAM):
            plot_spectrogram(ft, sr, axes)
        if (plot & Plot.SPECTRUM):
            plot_spectrum(ft, axes)
    return ft

def _expand_axes(axes):
    '''
    Expand axes so that they always are a 2D numpy array, independently of the number of plots.
    '''
    if not isinstance(axes, np.ndarray):
        return np.expand_dims(axes, axis=(0, 1))
    elif (len(axes.shape) == 1):
        return np.expand_dims(axes, axis=0)
    else:
        return axes

def show(x, outfile=None, plots=Plot.SPECTROGRAM | Plot.SPECTRUM | Plot.WAVE):
    '''
    Shows multiple plots describing each of the input audio sources
    :param x: array of audio sources to show.
    :type x: list of pairs (y, sr), where y is a float np.ndarray with shape (nsamples, nchannels) and
        sr is the sample rate (int).
    :param outfile: If 'None', plots will be shown as a Figure in the GUI. Otherwise, it's assumed to be the file path
        where the plots will be saved as an image.
    :type outfile: str
    :param plots: Flags indicating the plots to show for each audio source.
    :type plots: Plot
    '''
    num_plots = 0
    for plot in Plot:
        if (plots & plot):
            num_plots += 1

    num_srcs = len(x)
    fig, axes = plt.subplots(nrows=num_srcs, ncols=num_plots, sharex=False, sharey=False,
                             figsize=(num_plots*5, num_srcs*5))
    axes = _expand_axes(axes)
    for i in range(num_srcs):
        src = x[i]
        j = 0
        ft = None
        for plot in Plot:
            if (plots & plot):
                ax = axes[i][j]
                j += 1
                ft = _plot(plot, src, ax, ft)
    if (outfile is None):
        plt.show()
    else:
        plt.savefig(outfile)


if __name__ == '__main__':
    import librosa
    src1 = librosa.load(os.path.join(root_dir,"data/musdb/test/Al James - Schoolboy Facination.stem_vocals.wav"))
    src2 = librosa.load(os.path.join(root_dir,"data/musdb/test/Al James - Schoolboy Facination.stem_drums.wav"))
    show([src1,src2])
