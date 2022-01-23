import os, sys, argparse
import time
from functools import partial
import torch
import pickle
import numpy as np

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from torch.optim import Adam
from tqdm import tqdm

# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from model.model_params import ModelArgs
from model.waveunet_params import waveunet_params
import model.utils as model_utils
import utils
from data.dataset import SeparationDataset
from data.musdb import get_musdb_folds
from data.utils import crop_targets, random_amplify
from test import evaluate, validate
from model.waveunet import Waveunet
from math import ceil

def _create_waveunet(args):
    '''
    creates the waveunet model according to the given parameters
    @param args: the argument list of hyperparameters
    @return: waveunet model for audio source separation
    '''

    num_features = [args.features * i for i in range(1, args.levels + 1)] if args.feature_growth == "add" else \
        [args.features * 2 ** i for i in range(0, args.levels)]
    target_outputs = ceil(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, downsampling_kernel_size=args.downsampling_kernel_size,
                     upsampling_kernel_size=args.upsampling_kernel_size, bottleneck_kernel_size=args.bottleneck_kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate, num_convs=args.num_convs)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))
    return model

def _load_musdb(musdb, args, data_shapes, folds=None):
    '''
    loads data for testing, validation and training that is fitted to the shape of the model
    :param musdb: The musdb splits as returned by 'get_musdb_folds'.
    :param args: the argument list of hyperparameters
    :param data_shapes: model shape to fit data to model shape
    :return: source separation datasets for training, validation and testing as well as the dataloader for
    training dataset and the retrieved audio files
    :param folds: Used with kfold cross-validation. By default, musdb's official fixed train/validation split is used.
    If folds is given here, it's expected to be a dict whose 'train' entry contains the ids of the train samples,
    and its 'val' entry contains the validation id samples.
    '''

    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop_targets, shapes=data_shapes)
    # Data augmentation function for training
    augment_func = partial(random_amplify, shapes=data_shapes, min=0.7, max=1.0)
    train_data = SeparationDataset(musdb, "train", args.instruments, args.sr, args.channels, data_shapes, True,
                                   args.hdf_dir, audio_transform=augment_func)
    val_data = SeparationDataset(musdb, "val", args.instruments, args.sr, args.channels, data_shapes, False,
                                 args.hdf_dir, audio_transform=crop_func)
    train_subsampler = None
    val_subsampler = None
    train_val_shuffle = True
    if (folds is not None):
        # For k-fold cross validation concat train and val datasets in one and sample from the given folds instead.
        train_subsampler = torch.utils.data.SubsetRandomSampler(folds['train'])
        val_subsampler = torch.utils.data.SubsetRandomSampler(folds['val'])
        train_data = ConcatDataset([train_data, val_data])
        val_data = train_data
        train_val_shuffle = None

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=train_val_shuffle,
                                               num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn,
                                               sampler=train_subsampler)
    train_loader.num_samples = len(train_data)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=train_val_shuffle,
                                             num_workers=args.num_workers, sampler=val_subsampler)
    val_loader.num_samples = len(val_data)

    test_data = SeparationDataset(musdb, "test", args.instruments, args.sr, args.channels, data_shapes, False,
                                  args.hdf_dir, audio_transform=crop_func)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers)
    test_loader.num_samples = len(test_data)
    return train_loader, val_loader, test_loader

def _compute_metrics(args, musdb, model, writer, state):
    '''
    computes evaluation metrics and adds them to the tensorboard
    @param args: the argument list with hyperparameters
    @param musdb: musdb dataset object
    @param model: waveunet pytorch model
    @param writer: tensorboard object to log the metrics
    @param state: state with step value to record
    '''

    # Mir_eval metrics
    test_metrics = evaluate(args, musdb["test"], model, args.instruments)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(args.checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)

    # Write most important metrics into Tensorboard log
    avg_SDRs = {inst: np.mean([np.nanmean(song[inst]["SDR"]) for song in test_metrics]) for inst in args.instruments}
    avg_SIRs = {inst: np.mean([np.nanmean(song[inst]["SIR"]) for song in test_metrics]) for inst in args.instruments}
    for inst in args.instruments:
        sdr_name = "test_SDR_" + inst
        writer.add_scalar(sdr_name, avg_SDRs[inst], state["step"])
        print(f'{sdr_name}: {avg_SDRs[inst]}')
        sir_name = "test_SIR_" + inst
        writer.add_scalar(sir_name, avg_SIRs[inst], state["step"])
        print(f'{sir_name}: {avg_SIRs[inst]}')
    overall_SDR = np.mean([v for v in avg_SDRs.values()])
    writer.add_scalar("test_SDR", overall_SDR)
    print("SDR: " + str(overall_SDR))

def train_waveunet(args: argparse.Namespace, musdb, experiment_name: str = "exp", folds=None):
    '''
    Creates model from given hyperparameters, trains it to a given dataset, computes validation loss and performs testing
    :param args: argument list with hyperparmeters
    :param musdb: The musdb splits as returned by 'get_musdb_folds'.
    :param experiment_name: experiment name for checkpoint and log saving
    :param folds: Used with kfold cross-validation. By default, musdb's official fixed train/validation split is used.
    If folds is given here, it's expected to be a dict whose 'train' entry contains the ids of the train samples,
    and its 'val' entry contains the validation id samples.
    :return validation loss
    '''
    print(f'Start training {experiment_name}, args: {args}')

    # Create subdirectory for hdf intermediate format files with name <instruments>_<sr>_<channels>
    hdf_subdir = "_".join(args.instruments) + f"_{args.sr}_{args.channels}"
    args.hdf_dir = os.path.join(args.hdf_dir, hdf_subdir)

    # Save checkpoints and logs in separate directories for each experiment
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_name)
    args.log_dir = os.path.join(args.log_dir, experiment_name)

    model = _create_waveunet(args)
    writer = SummaryWriter(args.log_dir)

    train_loader, val_loader, test_loader = _load_musdb(musdb, args, model.shapes, folds)

    ##### TRAINING ####

    # Set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    # Set up training state dict that will also be saved into checkpoints
    state = {"step": 0,
             "worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)
    
    if not args.skip_training:
        print('TRAINING START')
        while state["worse_epochs"] < args.patience:
            print("Training one epoch from iteration " + str(state["step"]))
            avg_time = 0.
            model.train()
            num_it = train_loader.num_samples // args.batch_size
            with tqdm(total=num_it) as pbar:
                np.random.seed()
                for example_num, (x, targets) in enumerate(train_loader):
                    if args.cuda:
                        x = x.cuda()
                        for k in list(targets.keys()):
                            targets[k] = targets[k].cuda()

                    t = time.time()

                    # Set LR for this iteration
                    utils.set_cyclic_lr(optimizer, example_num, num_it, args.cycles,
                                        args.min_lr, args.lr)
                    writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                    # Compute loss for each instrument/model
                    optimizer.zero_grad()
                    if (args.separate == 0 and state["step"] == 0):
                        print("Saving model graph")
                        writer.add_graph(model, x, use_strict_trace=False)
                        print("Graph added to logs")
                    outputs, avg_loss = model_utils.compute_loss(model, x, targets, criterion, compute_grad=True)

                    optimizer.step()

                    state["step"] += 1

                    t = time.time() - t
                    avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                    writer.add_scalar("train_loss", avg_loss, state["step"])

                    if example_num % args.example_freq == 0:
                        input_centre = torch.mean(
                            x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]],
                            0)  # Stereo not supported for logs yet
                        writer.add_audio("input", input_centre, state["step"], sample_rate=args.sr)

                        for inst in outputs.keys():
                            writer.add_audio(inst + "_pred", torch.mean(outputs[inst][0], 0), state["step"],
                                             sample_rate=args.sr)
                            writer.add_audio(inst + "_target", torch.mean(targets[inst][0], 0), state["step"],
                                             sample_rate=args.sr)

                    pbar.update(1)
            # VALIDATE
            val_loss = validate(args, model, criterion, val_loader)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))
            writer.add_scalar("val_loss", val_loss, state["step"])

            # EARLY STOPPING CHECK
            checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_checkpoint"] = checkpoint_path

            # CHECKPOINT
            print("Saving model...")
            model_utils.save_model(model, optimizer, state, checkpoint_path)

            state["epochs"] += 1
    else:
        assert args.load_model is not None

    print(f'Best checkpoint: {state["best_checkpoint"]}, val loss: {state["best_loss"]}')

    if (folds is None):
        #### TESTING ####
        # Test loss
        print("TESTING")

        # Load best model based on validation loss
        state = model_utils.load_model(model, None, state["best_checkpoint"], args.cuda)
        test_loss = validate(args, model, criterion, test_loader)
        print("TEST FINISHED: LOSS: " + str(test_loss))
        writer.add_scalar("test_loss", test_loss, state["step"])

        _compute_metrics(args, musdb, model, writer, state)

    writer.close()
    return state['best_loss']

_methods = {'waveunet': [train_waveunet, waveunet_params]}

def _create_kfolds(k, n):
    indcs = np.arange(n)
    np.random.shuffle(indcs)
    indcs_val_mask = np.empty(n, dtype=bool)
    fold_size = n // k
    val_fold_start = 0
    kfolds = []
    for i in range(k):
        val_fold_end = val_fold_start + fold_size if i < k - 1 else n
        indcs_val_mask[:] = False
        indcs_val_mask[val_fold_start:val_fold_end] = True
        kfolds.append({'train': indcs[~indcs_val_mask], 'val': indcs[indcs_val_mask]})
        val_fold_start += fold_size
    return kfolds


def train_apply(method = 'waveunet', dataset = 'musdb', datasets_path='/home/space/datasets',
                model_args: ModelArgs = None, job_name = 'experiment',
                task_id = 0, task_index = 0, num_tasks = 1):
    '''
    performs training with hyperparameters. Uses default parameters if none are given
    :param method: the waveunet model
    :param dataset: original musdb dataset or the extended musdb dataset
    :param datasets_path: Root directory where datasets are located
    :param model_args: argument list with hyperparameters
    :param job_name: job name
    :param task_id: task id
    :param task_index: task index
    :param num_tasks: tasks number
    '''

    if method not in _methods:
        raise ValueError(f"Unknown method {method}.")

    train_func, model_params = _methods[method]

    if (model_args is None):
            # Use default waveunet params
            args = model_params.get_defaults()
            # Set dataset_dir and given args, fill in other parameters with default values.
            args.dataset_dir = os.path.join(datasets_path, dataset)
            musdb = get_musdb_folds(args.dataset_dir)
            train_func(args, musdb)
            return

    musdb = get_musdb_folds(model_args.get().dataset_dir)

    if (model_args.get_num_combs() == 1):
        # Experiment with fixed hyper-parameters, use musdb's fixed train val split.
        train_func(model_args.get(), musdb, experiment_name=f"job_{job_name}")
        return

    # Model evaluation: use K-fold cross validation.
    model_args = model_args.get_comb_partition(task_index, num_tasks)
    k = 5 # number of folds
    n = len(musdb['train']) + len(musdb['val'])
    min_loss = float('inf')
    best_args = None

    # For each hyperparam combination
    for i in range(model_args.get_num_combs()):
        args_comb = model_args.get_comb(i)
        avg_loss = 0
        # For each fold
        for fold_id, folds in enumerate(_create_kfolds(k, n)):
            print(f'Fold {fold_id}. Train: {folds["train"]}, val: {folds["val"]}')
            loss = train_func(args_comb, musdb, experiment_name=f"job_{job_name}_task{task_id}_exp{i}_fold{fold_id}", folds=folds)
            avg_loss += loss
        avg_loss /= k
        if (avg_loss < min_loss):
            min_loss = avg_loss
            best_args = args_comb

    print(f'Best model, args: {best_args}, loss: {min_loss}')


