#!/usr/bin/env bash
#$ -N name
# name of the experiment
#$ -l cuda=1
# remove this line when no GPU is needed!
#$ -q all.q
# do not fill the qlogin queue
#$ -cwd
# start processes in current directory
#$ -V
# provide environment variables
##$ -t 1-10 (commented out)
# start 10 instances: from 1 to 10

# Experiments from WaveUNet paper:
# M1
# Adam's Beta1 (0.9) and Beta2 (0.999) are Adam's default values.
# kernel size 15 for downsampling, 5 for upsampling
# Second training phase with batch size 32 and lr 1e-5
python train.py --hdf_dir=/home/space/datasets/musdb/hdf --cuda --instruments accompaniment vocals\
--cycles 1 --sr 22050 --channels 1 --output_size ? --patience 20 --separate 0 --features 24 --lr 1e-4 --min_lr 1e-4 \
--batch_size 16 --levels 6 --depth 1 --kernel_size ? --strides ? --loss L2 --conv_type normal --res fixed
--feature_growth add

# M4
#
# TODO: missing difference output layer
# default
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --cuda
