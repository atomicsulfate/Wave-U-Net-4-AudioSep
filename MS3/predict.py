import sys, os
# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from MS1.load_data import load_data
from utils import load_model, get_def_model_args

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

    return {}, 0

if __name__ == '__main__':
    train_set, test_set = load_data(datasets_path='data')
    sample = train_set[0][0]
    _, confidence = predict(sample, model=load_model(get_def_model_args(),'checkpoints/waveunet/best_checkpoint'))
    print(f'Confidence {confidence}')