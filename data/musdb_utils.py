import os
from data.utils import write_wav

def save_estimates(track_name, output_path, target_estimates, sample_rate=22050):
    track_estimates_dir = os.path.join(output_path, track_name)

    for target_name, estimate in target_estimates.items():
        estimate_path = os.path.join(track_estimates_dir, target_name + ".wav")
        os.makedirs(track_estimates_dir, exist_ok=True)
        write_wav(estimate_path, estimate, sample_rate)
