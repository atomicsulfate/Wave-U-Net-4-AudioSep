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
########################################################

# test with musdb M3 trained with musdb_extended
python test.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals \
--load_model=/home/pml_17/checkpoints/waveunet/job_M3_musdb_ext_acc_vocals_sr22050_mono_task0_exp0/checkpoint_765232 \
--cycles 1 --sr 22050 --channels 1 --output_size 0.743 --patience 20 --separate 0 --features 24 \
--lr 1e-4 --min_lr 1e-4 --batch_size 16 --levels 13 --depth 1 --downsampling_kernel_size 15 --bottleneck_kernel_size 15 \
--upsampling_kernel_size 5 --strides 2 --loss L2 --conv_type normal --res naive --feature_growth add --num_convs 1


#####################
# default
####################

# test with musdb default trained with musdb_extended
#python test.py --hdf_dir=/home/space/datasets/musdb/hdf --dataset_dir=/home/space/datasets/musdb --cuda --instruments accompaniment vocals \
#--load_model=/home/pml_17/checkpoints/waveunet/job_default_pytorch_musdb_ext_acc_vocals_sr22050_mono_task0_exp0/checkpoint_1091600 \
#--sr 22050 --channels 1
