from waveUNet.test import *
import os
import pickle
from torch.utils.tensorboard import SummaryWriter
from model.waveunet_params import waveunet_params
from model.waveunet import Waveunet
from data.musdb import get_musdb_folds
from math import ceil
from functools import partial
from data.dataset import SeparationDataset
from data.utils import crop_targets, random_amplify

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
        sdr_name = "test_SDR_" + inst
        writer.add_scalar(sdr_name, avg_SDRs[inst], state["step"])
        print(f'{sdr_name}: {avg_SDRs[inst]}')
        sir_name = "test_SIR_" + inst
        writer.add_scalar(sir_name, avg_SIRs[inst], state["step"])
        print(f'{sir_name}: {avg_SIRs[inst]}')
    overall_SDR = np.mean([v for v in avg_SDRs.values()])
    writer.add_scalar("test_SDR", overall_SDR)
    print("SDR: " + str(overall_SDR))

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

if __name__ == '__main__':
    experiment_name = os.environ['JOB_NAME'] if 'JOB_NAME' in os.environ else 'test'
    args = waveunet_params.parse_args().get_comb_partition(0, 1).get_comb(0)

    hdf_subdir = "_".join(args.instruments) + f"_{args.sr}_{args.channels}"
    args.hdf_dir = os.path.join(args.hdf_dir, hdf_subdir)

    if (not os.path.exists(args.dataset_dir)):
        raise ValueError(f"Dataset directory {args.dataset_dir} does not exist.")

    # Save checkpoints and logs in separate directories for each experiment
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_name)
    os.makedirs(args.checkpoint_dir,exist_ok=True)
    args.log_dir = os.path.join(args.log_dir, experiment_name)

    model = _create_waveunet(args)
    writer = SummaryWriter(args.log_dir)

    train_data, val_data, test_data, dataloader, musdb = _load_musdb(args, model.shapes)

    print("Test model from checkpoint " + str(args.load_model))
    state = model_utils.load_model(model, None, args.load_model, args.cuda)
    _compute_metrics(args, musdb, model, writer, state)
    writer.close()