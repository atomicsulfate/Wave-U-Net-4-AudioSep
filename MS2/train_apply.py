import os, sys, argparse
import time
from functools import partial
import torch
import pickle
import numpy as np

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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

def _load_musdb(args, data_shapes):
    musdb = get_musdb_folds(args.dataset_dir)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop_targets, shapes=data_shapes)
    # Data augmentation function for training
    augment_func = partial(random_amplify, shapes=data_shapes, min=0.7, max=1.0)
    train_data = SeparationDataset(musdb, "train", args.instruments, args.sr, args.channels, data_shapes, True,
                                   args.hdf_dir, audio_transform=augment_func)
    val_data = SeparationDataset(musdb, "val", args.instruments, args.sr, args.channels, data_shapes, False,
                                 args.hdf_dir, audio_transform=crop_func)
    test_data = SeparationDataset(musdb, "test", args.instruments, args.sr, args.channels, data_shapes, False,
                                  args.hdf_dir, audio_transform=crop_func)

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
    return train_data, val_data, test_data, dataloader, musdb

def _compute_metrics(args, musdb, model, writer, state):
    # Mir_eval metrics
    test_metrics = evaluate(args, musdb["test"], model, args.instruments)

    # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(args.checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)

    # Write most important metrics into Tensorboard log
    avg_SDRs = {inst: np.mean([np.nanmean(song[inst]["SDR"]) for song in test_metrics]) for inst in args.instruments}
    avg_SIRs = {inst: np.mean([np.nanmean(song[inst]["SIR"]) for song in test_metrics]) for inst in args.instruments}
    for inst in args.instruments:
        writer.add_scalar("test_SDR_" + inst, avg_SDRs[inst], state["step"])
        writer.add_scalar("test_SIR_" + inst, avg_SIRs[inst], state["step"])
    overall_SDR = np.mean([v for v in avg_SDRs.values()])
    writer.add_scalar("test_SDR", overall_SDR)
    print("SDR: " + str(overall_SDR))

def train_waveunet(args: argparse.Namespace, experiment_name: str = "exp"):
    # Create subdirectory for hdf intermediate format files with name <instruments>_<sr>_<channels>
    hdf_subdir = "_".join(args.instruments) + f"_{args.sr}_{args.channels}"
    args.hdf_dir = os.path.join(args.hdf_dir, hdf_subdir)

    if (not os.path.exists(args.dataset_dir)):
        raise ValueError(f"Dataset directory {args.dataset_dir} does not exist.")

    # Save checkpoints and logs in separate directories for each experiment
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_name)
    args.log_dir = os.path.join(args.log_dir, experiment_name)

    model = _create_waveunet(args)
    writer = SummaryWriter(args.log_dir)

    train_data, val_data, test_data, dataloader, musdb = _load_musdb(args, model.shapes)

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

    print('TRAINING START')
    while state["worse_epochs"] < args.patience:
        print("Training one epoch from iteration " + str(state["step"]))
        avg_time = 0.
        model.train()
        with tqdm(total=len(train_data) // args.batch_size) as pbar:
            np.random.seed()
            for example_num, (x, targets) in enumerate(dataloader):
                if args.cuda:
                    x = x.cuda()
                    for k in list(targets.keys()):
                        targets[k] = targets[k].cuda()

                t = time.time()

                # Set LR for this iteration
                utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles,
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
        val_loss = validate(args, model, criterion, val_data)
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

    #### TESTING ####
    # Test loss
    print("TESTING")

    # Load best model based on validation loss
    state = model_utils.load_model(model, None, state["best_checkpoint"], args.cuda)
    test_loss = validate(args, model, criterion, test_data)
    print("TEST FINISHED: LOSS: " + str(test_loss))
    writer.add_scalar("test_loss", test_loss, state["step"])

    _compute_metrics(args, musdb, model, writer, state)

    writer.close()

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
