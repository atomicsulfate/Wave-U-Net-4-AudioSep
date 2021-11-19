from enum import Flag, auto
import matplotlib.pyplot as plt
from data.audio_prepro import stft
from data.viz import plot_wave, plot_spectrogram, plot_spectrum
import numpy as np

class Plot(Flag):
    SPECTROGRAM = auto()
    SPECTRUM = auto()
    WAVE = auto()

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
    if isinstance(axes, np.ndarray):
        if (len(axes.shape) == 1):
            axes = np.expand_dims(axes, axis=0)
    else:
        axes = np.expand_dims(axes, axis=(0,1))
    for i in range(num_srcs):
        src = x[i]
        y,sr = src
        j = 0
        if (plots & Plot.WAVE):
            ax = axes[i][j]
            j+=1
            plot_wave(y,sr,ax)
        if (plots & (Plot.SPECTROGRAM | Plot.SPECTRUM)):
            ft = stft(y)
            if (plots & Plot.SPECTROGRAM):
                ax = axes[i][j]
                j+=1
                plot_spectrogram(ft, sr, ax)
            if (plots & Plot.SPECTRUM):
                ax = axes[i][j]
                j+=1
                plot_spectrum(ft, ax)

    plt.show()


if __name__ == '__main__':
    import librosa
    src1 = librosa.load("data/musdb/test/Al James - Schoolboy Facination.stem_vocals.wav")
    src2 = librosa.load("data/musdb/test/Al James - Schoolboy Facination.stem_drums.wav")
    show([src1,src2])
