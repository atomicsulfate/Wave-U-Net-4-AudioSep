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
#########################################
# M3
# TODO: output activation function tanh?
# TODO: missing difference output layer
# Uses conv with sinc lowpass filter with given stride instead of straight decimation.
# Uses transposed conv with sinc lowpass filter with given stride instead of linear interpolation.
# --output_size 0.743 seconds =>  0.743 * 22050 =16384 (output samples in paper).
########################################################

# with musdb
# 1st train
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals \
#--cycles 1 --sr 22050 --channels 1 --output_size 0.743 --patience 20 --separate 0 --features 24 \
#--lr 1e-4 --min_lr 1e-4 --batch_size 16 --levels 13 --depth 1 --downsampling_kernel_size 15 --bottleneck_kernel_size 15 \
#--upsampling_kernel_size 5 --strides 2 --loss L2 --conv_type normal --res naive --feature_growth add --num_convs 1

# 2nd train (refined)
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb \
#--load_model=/home/pml_17/checkpoints/waveunet/job_M3_musdb_acc_vocals_sr22050_mono_task0_exp0/checkpoint_98940 --cuda --instruments accompaniment vocals \
#--cycles 1 --sr 22050 --channels 1 --output_size 0.743 --patience 20 --separate 0 --features 24 \
#--lr 1e-5 --min_lr 1e-5 --batch_size 32 --levels 13 --depth 1 --downsampling_kernel_size 15 --bottleneck_kernel_size 15 \
#--upsampling_kernel_size 5 --strides 2 --loss L2 --conv_type normal --res naive --feature_growth add --num_convs 1

# with musdb_extended
# 1st train
#python train.py --hdf_dir=/home/space/datasets/musdb_extended/hdf --dataset_dir=/home/space/datasets/musdb_extended --cuda --instruments accompaniment vocals \
#--cycles 1 --sr 22050 --channels 1 --output_size 0.743 --patience 20 --separate 0 --features 24 \
#--lr 1e-4 --min_lr 1e-4 --batch_size 16 --levels 13 --depth 1 --downsampling_kernel_size 15 --bottleneck_kernel_size 15 \
#--upsampling_kernel_size 5 --strides 2 --loss L2 --conv_type normal --res naive --feature_growth add --num_convs 1

# test
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals \
#--load_model=/home/pml_17/checkpoints/waveunet/job_M3_musdb_ext_acc_vocals_sr22050_mono_task0_exp0/checkpoint_765232 --skip_training \
#--cycles 1 --sr 22050 --channels 1 --output_size 0.743 --patience 20 --separate 0 --features 24 \
#--lr 1e-4 --min_lr 1e-4 --batch_size 16 --levels 13 --depth 1 --downsampling_kernel_size 15 --bottleneck_kernel_size 15 \
#--upsampling_kernel_size 5 --strides 2 --loss L2 --conv_type normal --res naive --feature_growth add --num_convs 1


# 2nd train (refined)
#python train.py --hdf_dir=/home/space/datasets/musdb_extended/hdf --dataset_dir=/home/space/datasets/musdb_extended \
#--load_model=/home/pml_17/checkpoints/waveunet/job_M3_musdb_acc_vocals_sr22050_mono_task0_exp0/checkpoint_98940 --cuda --instruments accompaniment vocals \
#--cycles 1 --sr 22050 --channels 1 --output_size 0.743 --patience 20 --separate 0 --features 24 \
#--lr 1e-5 --min_lr 1e-5 --batch_size 32 --levels 13 --depth 1 --downsampling_kernel_size 15 --bottleneck_kernel_size 15 \
#--upsampling_kernel_size 5 --strides 2 --loss L2 --conv_type normal --res naive --feature_growth add --num_convs 1


#####################
# default
####################
# with musdb
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals --sr 22050 --channels 1

# with musdb_extended
#python train.py --hdf_dir=/home/space/datasets/musdb_extended/hdf --dataset_dir=/home/space/datasets/musdb_extended --cuda --instruments accompaniment vocals --sr 22050 --channels 1

# test with musdb default trained with musdb_extended
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals \
#--load_model=/home/pml_17/checkpoints/waveunet/job_default_pytorch_musdb_ext_acc_vocals_sr22050_mono_task0_exp0/checkpoint_1091600 \
#--sr 22050 --channels 1

#####################
# Model selection
####################

# 1) Learning params (x9)
python train.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals --sr 22050 --channels 1 --patience 8 \
--features 24 --levels 6 --depth 1 --loss L2 --num_convs 2 --res fixed --cycles 2 --conv_type normal \
--lr 1e-3 1e-4 1e-5 --min_lr 1e-6 --batch_size 8 16 32

# 2) Normalization (Opt) -> Choose gn for batch_size <=16, bn for 32.
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals --sr 22050 --channels 1 --patience 8 \
#--features 24 --levels 6 --depth 1 --loss L2 --num_convs 2 --res fixed --cycles 2 --lr X --min_lr 1e-6 --batch_size X \
#--conv_type normal gn bn

# 3) Resampling (x3)
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals --sr 22050 --channels 1 --patience 8 \
#--features 24 --levels 6 --depth 1 --loss L2 --num_convs 2 --cycles 2 --lr X --min_lr 1e-6 --batch_size X --conv_type X \
#--res naive fixed learned

# 3) Model size (x8)
#python train.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals --sr 22050 --channels 1 --patience 8 \
#--loss L2 --num_convs 2 --res fixed --cycles 2 --lr X --min_lr 1e-6 --batch_size X --conv_type X --res X \
#--features 24 32 --levels 6 12 --depth 1 2
