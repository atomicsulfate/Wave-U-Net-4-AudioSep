import sys, os
# add project root dir to sys.path so that all packages can be found by python.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import model.utils as model_utils
from model.waveunet_params import waveunet_params
from model.waveunet import Waveunet
from math import ceil

def create_waveunet(args, log=True):
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
        if (log):
            print("move model to gpu")
        model.cuda()

    if (log):
        print('model: ', model)
        print('parameter count: ', str(sum(p.numel() for p in model.parameters())))
    return model

def load_model(args, checkpoint):
    model = create_waveunet(args)
    model_utils.load_model(model=model, optimizer=None, path=checkpoint, cuda=False)
    return model

def load_default_model():
    args = waveunet_params.get_defaults()
    args.instruments = ["accompaniment", "vocals"]
    args.sr = 22050
    args.channels = 1
    return load_model(args, "MS3/checkpoint_493495")