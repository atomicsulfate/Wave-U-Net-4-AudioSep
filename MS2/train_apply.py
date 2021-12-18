import os, sys, argparse
# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
import waveUNet.train
from model.model_params import ModelArgs
from model.waveunet import waveunet_params

def train_waveunet(args: argparse.Namespace, experiment_name: str = "exp"):
    # Create subdirectory for hdf intermediate format files with name <instruments>_<sr>_<channels>
    hdf_subdir = "_".join(args.instruments) + f"_{args.sr}_{args.channels}"
    args.hdf_dir = os.path.join(args.hdf_dir, hdf_subdir)

    if (not os.path.exists(args.dataset_dir)):
        raise ValueError(f"Dataset directory {args.dataset_dir} does not exist.")

    # Save checkpoints and logs in separate directories for each experiment
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_name)
    args.log_dir = os.path.join(args.log_dir, experiment_name)

    waveUNet.train.main(args)

_methods = {'waveunet': [train_waveunet, waveunet_params]}

def train_apply(method = 'waveunet', dataset = 'musdb', datasets_path='/home/space/datasets',
                model_args: ModelArgs = None, job_name = 'experiment',
                task_id = 0, task_index = 0, num_tasks = 1):
    if method not in _methods:
        raise ValueError(f"Unknown method {method}.")

    train_func, model_params = _methods[method]

    if (model_args is None):
            # Use default waveunet params
            args = model_params.get_defaults()
            # Set dataset_dir and given args, fill in other parameters with default values.
            args.dataset_dir = os.path.join(datasets_path, dataset)
            train_func(args)
            return

    model_args = model_args.get_comb_partition(task_index, num_tasks)

    for i in range(model_args.get_num_combs()):
        args_comb = model_args.get_comb(i)
        print(f'Start training for job {job_name}, task {task_id}, params {i}: {args_comb}')
        train_func(args_comb, experiment_name=f"job_{job_name}_task{task_id}_exp{i}")


