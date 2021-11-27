import os
import museval
from pathlib import Path
import simplejson
import numpy as np
import musdb
from tqdm import tqdm

def evaluate_estimates(db, estimates_dir):
    museval.eval_mus_dir(db, estimates_dir, output_dir=estimates_dir)

def evaluate_track_estimates(track_name, target_refs, target_estimates, estimates_dir, sample_rate=22050):
    class DummyTarget:
        def __init__(self, audio):
            self.audio = audio

    track = musdb.MultiTrack("", name=track_name,subset='test', targets=None, sample_rate=sample_rate)
    track.targets = { track_name: DummyTarget(audio) for track_name, audio in target_refs.items()}
    track.rate = sample_rate

    museval.eval_mus_track(track, target_estimates, estimates_dir)

def create_eval_store(estimates_dir):
    p = Path(estimates_dir)
    if not p.exists():
        return None
    store = museval.EvalStore()
    json_paths = p.glob('**/*.json')
    for json_path in json_paths:
        with open(json_path) as json_file:
            json_string = simplejson.loads(json_file.read())
        track_df = museval.json2df(json_string, json_path.stem)
        store.add_track(track_df)
    return store

def create_method_store(root_estimates_dir, method_dirs, subset=None):
    store = museval.MethodStore()
    for method in tqdm(method_dirs,"Creating eval store"):
        method_estimates_dir = os.path.join(root_estimates_dir, method)
        if (subset is not None):
            method_estimates_dir = os.path.join(method_estimates_dir, subset)
        store.add_evalstore(create_eval_store(method_estimates_dir), method)

    return store

def _filter_df_col(df, col_name, col_values):
    if (col_values is not None):
        col_values = [col_values] if isinstance(col_values, str) else col_values
        df = df[df[col_name].isin(col_values)]
    if (len(df[col_name].unique()) == 1):
        df = df.drop(col_name, axis=1)
    return df

def plot_violin(method_store, axes, methods=None, metrics=None, targets=None, title="Metrics"):
    '''
    :param method_store: Store with evaluation metrics
    :type method_store: museval.MethodStore
    '''

    # Get median scores for each track
    df = method_store.agg_frames_scores().reset_index(name='score')

    # Get lists of scores per method-metric-target
    df = df.groupby(['method','metric','target'])['score'].apply(list).reset_index(name='score')

    # filter by provided column values
    df = _filter_df_col( _filter_df_col( _filter_df_col(df, 'method', methods),'metric', metrics), 'target', targets)

    values = df['score'].values
    xtick_labels = df.drop('score', axis=1)
    xlabel = ','.join(xtick_labels.columns.values)
    xtick_labels = xtick_labels.apply(','.join, axis=1).values

    axes.violinplot(values, showmedians=True)
    axes.set_title(title)
    axes.set_ylabel("Score")
    axes.set_xlabel(xlabel)
    axes.set_xticks(np.arange(1, len(xtick_labels) + 1))
    axes.set_xticklabels(xtick_labels)

