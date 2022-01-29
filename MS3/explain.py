import sys, os
import numpy as np
import math
# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from MS1.load_data import load_data
from utils import load_model, load_default_model

def explain(x, sources = ["accompaniment", "vocals"], resolution=10, sr=22050, model=load_default_model()):
    '''
    Returns an explanation of the predicted separation of an input song into its sources or instruments.
    :param x: An audio sample for which the source separation must be predicted, as returned
    by the function 'load_data' (see Milestone 1).
    :param sources: Audio sources to predict ("accompaniment", "vocals", "bass", "drums", "other").
    :param resolution: Resolution of the explanation in milliseconds.
    :param sr: Sampling rate of the input.
    :param model: The trained model to be used for the prediction.
    :returns: An 3D array of size (n,n,s), with n the number of audio windows per source at the given time resolution and
    s the number of sources.
    Element (i,j,k) is a normalized value [0-1] of the importance that input window i has on the prediction of output
    window j for source k.
    '''
    # channels = x.shape[0]
    n = math.ceil(1e3 * x.shape[1] / (resolution * sr))
    s = len(sources)
    return np.zeros([n,n,s]) # <- might make sense to make this a sparse matrix instead, otherwise it's huge.

if __name__ == '__main__':
    train_set, test_set = load_data(datasets_path='data')
    sample = train_set[0][0]
    explanation = explain(sample)
    print(f'Sample {sample.shape}, explanation {explanation.shape}')