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
# M3
# TODO: output activation function tanh?
# TODO: missing difference output layer
# Uses conv with sinc lowpass filter with given stride instead of straight decimation.
# Uses transposed conv with sinc lowpass filter with given stride instead of linear interpolation.
# --output_size 0.743 seconds =>  0.743 * 22050 =16384 (output samples in paper).
python train.py --hdf_dir=/home/space/datasets/musdb/hdf --cuda --instruments accompaniment vocals\
--cycles 1 --sr 22050 --channels 1 --output_size 0.743 --patience 20 --separate 0 --features 24\
--lr 1e-4 --min_lr 1e-4 --batch_size 16 --levels 13 --depth 1 --downsampling_kernel_size 15 --bottleneck_kernel_size 15\
--upsampling_kernel_size 5 --strides 2 --loss L2 --conv_type normal --res naive --feature_growth add --num_convs 1

# default
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --cuda --instruments accompaniment vocals --sr 22050 --channels 1
