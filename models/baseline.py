import sys,os

# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import numpy as np
from scipy.signal import stft, istft
from data.musdb_loader import get_musdb_folds
from data.dataset import SeparationDataset
from data.eval import evaluate_track_estimates
from tqdm import tqdm
from data.musdb_utils import save_estimates

class BaselineModel:
    def __init__(self, mask_type="binary", alpha=1,
                 thetas = {'vocals':0.35, 'drums':0.5, 'bass':0.5, 'other':0.6, 'accompaniment': 0.8},
                 nfft = 2048):
        self.alpha = alpha
        self.nfft = nfft
        self.target_masks = {}
        # small epsilon to avoid dividing by zero
        self.eps = np.finfo(np.float64).eps
        self.mask_type = mask_type
        self.thetas = thetas

    def _compute_target_spectrograms(self, targets):
        # Compute targets spectrograms
        spectrograms = {}
        for name, target in targets.items():
            # compute spectrogram of target source:
            # magnitude of STFT to the power alpha
            spectrograms[name] = np.abs(stft(target, nperseg=self.nfft)[-1]) ** self.alpha
        return spectrograms

    def _compute_mix_spectrogram(self, target_spectrograms):
        # compute model as the sum of spectrograms
        model = 0
        for target_spectrogram in target_spectrograms.values():
            model += target_spectrogram
        return model

    def _compute_target_ratio_masks(self, mix_spectrogram, target_spectrograms):
        # Add eps to the mix spectrogram to avoid divide by 0
        mix_spectrogram += self.eps

        accompaniment_mask = 0
        norm = 0
        for name, target_spectrogram in target_spectrograms.items():
            # compute soft mask as the ratio between source spectrogram and total average across time windows
            target_mask = np.mean(np.divide(np.abs(target_spectrogram), mix_spectrogram), axis=2)[:, :, np.newaxis]
            self.target_masks[name] = target_mask
            norm += target_mask
            # accumulate to the accompaniment if this is not vocals
            if name != 'vocals':
                accompaniment_mask += target_mask

        self.target_masks['accompaniment'] = accompaniment_mask
        # normalize
        for target_name in self.target_masks.keys():
            self.target_masks[target_name] = np.divide(self.target_masks[target_name], norm)

    def _fit_ratio_masks(self, targets):
        target_spectograms = self._compute_target_spectrograms(targets)
        mix_spectrogram = self._compute_mix_spectrogram(target_spectograms)
        self._compute_target_ratio_masks(mix_spectrogram, target_spectograms)

    def _fit_binary_masks(self, targets):
        self._fit_ratio_masks(targets)

        for target_name, target_mask in self.target_masks.items():
            theta = self.thetas[target_name]
            target_mask[np.where(target_mask >= theta)] = 1
            target_mask[np.where(target_mask < theta)] = 0

    def fit(self, train_dataset):
        for _,targets in tqdm(train_dataset, "Fitting"):
            if ('accompaniment' in targets):
                del targets['accompaniment']

            if (self.mask_type == "ratio"):
                self._fit_ratio_masks(targets)
            else:
                self._fit_binary_masks(targets)

    def predict(self, mix, target_names):
        N = mix.shape[1]  # remember number of samples for future use

        X = stft(mix, nperseg=self.nfft)[-1]

        target_estimates = {}
        for target_name in target_names:
            mask = self.target_masks[target_name]
            # multiply the mix by the target mask
            Yj = np.multiply(X, mask)

            # invert to time domain
            estimate = istft(Yj)[1].T[:N, :]
            # Add eps to estimate to avoid totally silent targets (eval metrics fail on silent estimates)
            target_estimates[target_name] = estimate + self.eps
        return target_estimates

if __name__ == '__main__':
    db_path = os.path.join(root_dir,sys.argv[1])
    output_path = os.path.join(root_dir,sys.argv[2])  #'data/musdb/estimates/baseline'
    train_dataset = SeparationDataset(get_musdb_folds(db_path)['train'])

    model = BaselineModel()
    model.fit(train_dataset)

    test_paths = get_musdb_folds(db_path)['test']
    test_dataset = SeparationDataset(test_paths)
    track_names = list(map(lambda target_dict: os.path.basename(os.path.dirname(target_dict['mix'])), test_paths))
    target_names = list(test_paths[0].keys())
    target_names.remove('mix')
    target_estimates = []
    target_references = []
    for mix, targets in tqdm(test_dataset, "Predicting"):
        target_references.append(targets)
        target_estimates.append(model.predict(mix, target_names))

    output_test_path = os.path.join(output_path, 'test')
    for i, track_estimates in enumerate(tqdm(target_estimates, "Saving estimates and evaluations")):
        track_name = track_names[i]
        track_references = target_references[i]
        save_estimates(track_name, output_test_path, target_estimates)
        for target_name in target_names:
            track_references[target_name] = track_references[target_name].numpy().T
        evaluate_track_estimates(track_name, track_references, track_estimates, output_path)


