import sys, os
# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from MS1.load_data import load_data
from utils import load_model, get_def_model_args
import torch
from torch.nn.functional import conv1d
import numpy as np

def_window_secs = 2
def_hop_secs = 1.5

def _predict_audio(audio, model):
    mix_audio = audio
    mix_len = mix_audio.shape[1]

    sources = predict(mix_audio, model)
    # In case we had to pad the mixture at the end, or we have a few samples too many due to inconsistent down- and upsamṕling, remove those samples from source prediction now
    for key in sources.keys():
        diff = sources[key].shape[1] - mix_len
        if diff > 0:
            print("WARNING: Cropping " + str(diff) + " samples")
            sources[key] = sources[key][:, :-diff]
        elif diff < 0:
            print("WARNING: Padding output by " + str(diff) + " samples")
            sources[key] = np.pad(sources[key], [(0, 0), (0, -diff)], "constant", 0.0)

    return sources

def _compute_residue(mixture_audio, vocals_audio, accompaniment_audio, sr,
                 window_secs = def_window_secs, hop_secs = def_hop_secs):
    # Residual energy after substracting predicted sources
    residual = (mixture_audio - (vocals_audio + accompaniment_audio))**2
    conv_input = torch.from_numpy(np.expand_dims(residual,0)).float()
    conv_kernel = torch.ones([1,1,round(window_secs*sr)])
    residual_t = conv1d(conv_input,conv_kernel,
                        stride=round(hop_secs*sr), padding='valid').squeeze()
    return residual_t

def predict(x, sources = ["accompaniment", "vocals"], sr=22050, model=None):
    '''
    Predicts the separation of a given song into the specified sources (i.e. "instruments"), specifying a
    level of confidence in the prediction.
    :param x: An audio sample for which the source separation must be predicted, as returned
    by the function 'load_data' (see Milestone 1).
    :param sources: Audio sources to predict ("accompaniment", "vocals", "bass", "drums", "other").
    :param sr: Sampling rate of the input.
    :param model: The trained model to be used for the prediction. If none passed, a default will be used.
    :returns: A pair whose first element is the predicted separation, and second is a normalized measurement (0-1) for
    the confidence in the given prediction.
    The prediction is a dictionary where each key is the source name and the value is an array containing the predicted
    audio time series.
    '''

    if (model is None):
        model = load_model()
    mix, _ = x
    prediction = _predict_audio(mix.detach().cpu().numpy(), model)
    residual = _compute_residue(mix, prediction["vocals"], prediction["accompaniment"], sr)
    return residual

    return prediction, 0

if __name__ == '__main__':
    train_set, test_set = load_data(datasets_path='data')
    sample = train_set[0][0]
    _, confidence = predict(sample, model=load_model(get_def_model_args(),'checkpoints/waveunet/best_checkpoint'))
    print(f'Confidence {confidence}')