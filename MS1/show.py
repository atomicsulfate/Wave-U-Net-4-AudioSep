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
from data.eval import create_method_store, plot_violin
import museval


class Plot(Flag):
    WAVE = auto()
    SPECTRUM = auto()
    SPECTROGRAM = auto()
    TRACK_SEP_METRICS = auto()
    GLOBAL_SEP_METRICS = auto()

def _plot(plot, y, sample_rate, axes, metrics_store=None, ft=None, signal_name=None):
    '''
    Call the plotting function corresponding to the given plot type.
    :returns: The fourier transform of source if it was computed or given as argument.
    '''
    suffix = f" ({signal_name})" if signal_name is not None else ""
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
    #if (plot & Plot.TRACK_SEP_METRICS and metrics_store is not None):
        # TODO
    return ft

def _show_signal(source, sample_rate, axes, plots, name=None, metrics_store=None):
    j = 0
    ft = None
    for plot in Plot:
        if (plots & plot):
            ax = axes[j]
            j += 1
            ft = _plot(plot, source, sample_rate, ax, metrics_store, ft, name)

def show(x=None, outfile=None, names=None, targets=None, metrics_store=None,
         plots=Plot.SPECTROGRAM | Plot.SPECTRUM | Plot.WAVE, sample_rate=22050, dpi=300):
    '''
    Shows multiple plots describing each of the input audio signals.
    :param x: array of audio signals to show.
    :type x: list[torch.Tensor or np.array], each element with shape (1,nsamples) (mono)
    :param outfile: If 'None', plots will be shown as a Figure in the GUI. Otherwise, it's assumed to be the file path
        where the plots will be saved as an image.
    :type outfile: str
    :param names: Optional list of names assigned to the audio signals in x (used for plot titles). By default, names
        will be 'Audio {i}' with {i} being the index of the audio signal in x.
    :param targets: If not 'None', it must be a list of the same length as x, with elements being dictionaries whose keys are
    strings, and values are the same type as x's elements. In this case, every element in x is assumed to
    be a mixture of multiple audio sources, and its corresponding dictionary in targets has the separated sources.
    :param metrics_store: Optional museval's MethodStore instance with separation performance metrics for the given dataset.
    :type metrics_store: museval.MethodStore
    :param plots: Flags indicating the plots to show for each audio source.
    :type plots: Plot
    :param sample_rate: Sample rate for all audio signals (default 22050)
    :type sample_rate: int
    :param dpi: Resolution used to draw matplotlib's Figure (default 100)
    :type dpi: int
    '''

    signal_plots = plots & ~Plot.GLOBAL_SEP_METRICS
    num_signal_plots = 0
    for plot in Plot:
        if (signal_plots & plot):
            num_signal_plots += 1

    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    num_signals = 0 if x is None else len(x)
    if (targets is not None and num_signals > 0 and len(targets) != num_signals):
        raise ValueError("targets must have same length as x")
    if (names is not None and num_signals > 0 and len(names) != num_signals):
        raise ValueError("names must have same length as x")
    num_figures = num_signals if num_signal_plots > 0 else 0
    if (plots & Plot.GLOBAL_SEP_METRICS):
        num_figures += 1
    subfigs = np.squeeze(plt.figure(constrained_layout=True, dpi=dpi).subfigures(nrows=num_figures, ncols=1, squeeze=False), axis=1)
    track_figs = subfigs[:-1] if (plots & Plot.GLOBAL_SEP_METRICS) else subfigs
    for signal_idx, subfig in enumerate(track_figs):
        signal = x[signal_idx]
        subfig_title = f'Audio {signal_idx}' if names is None else names[signal_idx]
        subfig.suptitle(subfig_title)
        num_targets = len(targets[signal_idx]) if targets is not None else 0
        axes = subfig.subplots(nrows=num_targets+1, ncols=num_signal_plots, squeeze=False, sharex=False, sharey=False)
        _show_signal(signal, sample_rate, axes[0], signal_plots, None if targets is None else "mix")
        if (targets is None):
            continue
        for src_idx,src in enumerate(targets[signal_idx].items()):
            src_name = src[0]
            src_signal = src[1]
            _show_signal(src_signal, sample_rate, axes[src_idx+1], signal_plots, src_name, metrics_store)

    if (plots & Plot.GLOBAL_SEP_METRICS):
        subfig = subfigs[-1]
        axes = subfig.subplots(1)
        target = metrics_store.df['target'].unique()[0]
        metric = metrics_store.df['metric'].unique()[0]
        plot_violin(metrics_store, axes, title=f"{target} | {metric}")
    if (outfile is None):
        plt.show()
    else:
        plt.savefig(outfile)

if __name__ == '__main__':
    import itertools

    dataset = 'musdb_extended'
    db_path = os.path.join(root_dir,'data', dataset)
    data = get_musdb_folds(db_path)
    path_list = data['test']
    track_names = list(map(lambda target_dict: os.path.basename(os.path.dirname(target_dict['mix'])), path_list))
    test = SeparationDataset(path_list)
    # mix, targets = test[0]
    # name = os.path.basename(os.path.dirname(path_list[0]['mix']))
    # voice_inst_targets = { 'vocals': targets['vocals'], 'accompaniment': targets['accompaniment']}

    #mixes, targets = map(list,zip(*test))
    #targets = list(map(lambda t:  { 'vocals': t['vocals'], 'accompaniment': t['accompaniment']}, targets))
    #show(mixes, outfile=None, names=names, targets=targets, metrics_store=store)


    # Global separation metrics
    store = create_method_store(os.path.join(db_path,'estimates'), ['baseline', 'ibm','irm'], 'test')
    df = store.df[store.df.track.isin(track_names)]

    for target, metric in itertools.product(['vocals', 'drums', 'bass', 'other', 'accompaniment'],
                                            ['SDR','ISR','SIR','SAR' ]):
        store.df = df[(df.metric == metric) & (df.target ==target)]
        show(outfile=f"plots/{dataset}_{target}_{metric}.png", metrics_store=store, plots=Plot.GLOBAL_SEP_METRICS)



