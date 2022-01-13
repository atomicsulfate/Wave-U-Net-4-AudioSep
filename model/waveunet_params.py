from model.model_params import ModelParams

waveunet_params = ModelParams()
waveunet_params.add_param('instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"],
               help="List of instruments to separate (default: \"bass drums other vocals\")")\
    .add_param('cuda', action='store_true', help='Use CUDA (default: False)')\
    .add_param('num_workers', type=int, default=1, help='Number of data loader worker threads (default: 1)')\
    .add_param('log_dir', type=str, default='logs/waveunet', help='Folder to write logs into')\
    .add_param('dataset_dir', type=str, default="/home/space/datasets/musdb", help='Dataset path')\
    .add_param('hdf_dir', type=str, default="hdf", help='Dataset path')\
    .add_param('checkpoint_dir', type=str, default='checkpoints/waveunet',help='Folder to write checkpoints into')\
    .add_param('load_model', type=str, default=None, help='Reload a previously trained model (whole task model)') \
    .add_param('cycles', type=int, default=2, help='Number of LR cycles per epoch') \
    .add_param('sr', type=int, default=44100, help="Sampling rate") \
    .add_param('channels', type=int, default=2, help="Number of input audio channels") \
    .add_param('output_size', type=float, default=2.0, help="Output duration") \
    .add_param('patience', type=int, default=20, help="Patience for early stopping on validation set") \
    .add_param('example_freq', type=int, default=200,
                  help="Write an audio summary into Tensorboard logs every X training iterations") \
    .add_param('separate', type=int, default=1,
                  help="Train separate model for each source (1) or only one (0)")\
    .add_hyperparam('features', type=int, default=32, help='Number of feature channels per layer')\
    .add_hyperparam('lr', type=float, default=1e-3, help='Initial learning rate in LR cycle (default: 1e-3)')\
    .add_hyperparam('min_lr', type=float, default=5e-5, help='Minimum learning rate in LR cycle (default: 5e-5)')\
    .add_hyperparam('batch_size', type=int, default=4, help="Batch size")\
    .add_hyperparam('levels', type=int, default=6, help="Number of DS/US blocks")\
    .add_hyperparam('depth', type=int, default=1, help="Number of convs per block")\
    .add_hyperparam('upsampling_kernel_size', type=int, default=5,
                  help="Filter width of kernels. Has to be an odd number")\
    .add_hyperparam('downsampling_kernel_size', type=int, default=5,
                  help="Filter width of kernels. Has to be an odd number")\
    .add_hyperparam('bottleneck_kernel_size', type=int, default=5,
                  help="Filter width of kernels. Has to be an odd number")\
    .add_hyperparam('strides', type=int, default=4, help="Strides in Waveunet")\
    .add_hyperparam('loss', type=str, default="L1", help="L1 or L2")\
    .add_hyperparam('conv_type', type=str, default="gn",
                  help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")\
    .add_hyperparam('num_convs', type=int, default=2,
                  help="Num convolutions to have, default=2, in the original paper it was 1")\
    .add_hyperparam('res', type=str, default="fixed",
                  help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned/naive")\
    .add_hyperparam('feature_growth', type=str, default="double",
                  help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")\
    .add_hyperparam('skip_training', type=bool, default=True,
                  help="Skip training to directly evaluate the checkpoint: True/False")
