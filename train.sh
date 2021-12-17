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
#$ -t 0-3
# start 4 instances: from 0 to 3

python train.py --hdf_dir=/home/space/datasets/musdb/hdf --cuda