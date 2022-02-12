import sys, os
import matplotlib.pyplot as plt

# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from MS1.load_data import load_data
from utils import load_model, get_def_model_args
import torch
from torch.nn.functional import conv1d
import numpy as np
from test import predict as test_predict
from data.viz import plot_prediction_confidence
from data.musdb_loader import get_musdb_folds
from data.dataset import RawSeparationDataset
from tqdm import tqdm

def _predict_audio(mix_audio, model):
    '''
    Predicts the seperation of the input audio mixture into its sources using a given model.
    :param mix_audio: The input audio mixture.
    :param model: The separation model.
    :returns: A dictionary of all predicted source signals.
    '''
    mix_len = mix_audio.shape[1]

    sources = test_predict(mix_audio, model)
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

def_window_secs = 2
def_hop_secs = 2

def _compute_snr(input_mix, sep_prediction, sr, window_secs = def_window_secs, hop_secs = def_hop_secs):
    '''
    Computes Signal-To-Noise-Ratio, where the signal is an input audio mixture and the noise is the difference
    between it and the sum of all predicted separated sources. The SNR is computed in windows, with each window having
    the specified size and being at the given hop distance from the next.
    :param input_mix The input mixture audio
    :param sep_prediction Predicted audio sources in a dictionary, where the keys are the source names and the values
           the corresponding audio time series.
    :param sr Sampling rate of all audio data.
    :param window_secs Size of each window in seconds.
    :param hop_secs Distance in seconds from a the start of a window and the start of the next one.
    :returns: An np.array with an SNR float number for each window.
    '''
    prediction_mix = sum(sep_prediction.values())
    noise_energy = (input_mix - prediction_mix) ** 2
    signal_energy = input_mix ** 2
    conv_kernel = torch.ones([1,1,round(window_secs*sr)])

    signal_power = conv1d(torch.from_numpy(np.expand_dims(signal_energy,0)).float(),conv_kernel,
                         stride=round(hop_secs*sr), padding='valid').squeeze()
    noise_power = conv1d(torch.from_numpy(np.expand_dims(noise_energy,0)).float(),conv_kernel,
                         stride=round(hop_secs*sr), padding='valid').squeeze()
    epsilon = 1e-6
    snr = 10*(np.log10(signal_power+epsilon) - np.log10(noise_power+epsilon))
    return snr

def predict(x, sr=22050, model=None, plot=None):
    '''
    Predicts the separation of a given song into the specified sources (i.e. "instruments"), specifying a
    level of confidence in the prediction.
    :param x: An audio sample for which the source separation must be predicted, as returned
    by the function 'load_data' (see Milestone 1).
    :param sr: Sampling rate of the input.
    :param model: The trained model to be used for the prediction. If none passed, a default will be used.
    :param plot: If set to 'True', it'll display the waveform plots of the input audio and the estimated
    separations colored by the confidence values. If it's set to a string, it'll be taken as the path to an image file
    where the plots with be saved. By default no plot is generated.
    :returns: A pair whose first element is the predicted separation, and second is a normalized measurement (0-1) for
    the confidence in the given prediction.
    The prediction is a dictionary where each key is the source name and the value is an array containing the predicted
    audio time series.
    '''

    if (model is None):
        model = load_model()
    prediction = _predict_audio(x.detach().cpu().numpy(), model)
    snr = _compute_snr(x, prediction, sr)

    # Use a logistic sigmoid to normalize SNR values to range [0,1]:
    # confidence = 1/(1 + e^(-k(x - mid_snr))).
    # Steepness and midpoint are empirically chosen based on results from MUSDB dataset.
    k = 0.15 # Steepness of the curve.
    mid_snr = 15 # SNR (dB) at which the confidence is 0.5.
    confidence =  1. / (1. + np.exp(-k*(snr - mid_snr)))

    if (plot == True):
        plot_prediction_confidence(x, prediction, confidence, sr)
        plt.show()
    elif (isinstance(plot,str)):
        plot_prediction_confidence(x, prediction, confidence, sr)
        plt.savefig(plot)
    return prediction, confidence

if __name__ == '__main__':
    train_set, test_set = load_data(datasets_path='data')
    sample = train_set[0][0]
    _, confidence = predict(sample, model=load_model(get_def_model_args(),'checkpoints/waveunet/best_checkpoint'), plot=True)
    print(f'Confidence {confidence}')

    # dataset = 'musdb'
    # db_path = os.path.join(root_dir, 'data', dataset)
    # data = get_musdb_folds(db_path)
    # path_list = data['test']
    # track_names = list(map(lambda target_dict: os.path.basename(os.path.dirname(target_dict['mix'])), path_list))
    # test = RawSeparationDataset(path_list)
    # model = load_model(get_def_model_args(), 'checkpoints/waveunet/best_checkpoint')
    # for i, sample in tqdm(enumerate(test)):
    #     predict(sample, model=model, plot=f'plots/vocals_conf_{track_names[i]}.png')
