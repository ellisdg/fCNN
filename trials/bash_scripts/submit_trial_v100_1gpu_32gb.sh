#!/bin/sh
#SBATCH --time=7-00:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=V100_2GPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_v100
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load cuda
module load anaconda
conda activate pytorch-1.5

export PYTHONPATH=/home/aizenberg/dgellis/fCNN:/home/aizenberg/dgellis/3DUnetCNN:${PYTHONPATH}

TRIAL=${1}
CONFIG=/home/aizenberg/dgellis/fCNN/data/${TRIAL}_config.json
HCC_CONFIG=/home/aizenberg/dgellis/fCNN/data/hcc_v100_1gpu_config.json
MODEL=/work/aizenberg/dgellis/fCNN/model_${TRIAL}.h5
LOG=/work/aizenberg/dgellis/fCNN/log_${TRIAL}.csv

/home/aizenberg/dgellis/.conda/envs/pytorch-1.5/bin/python /home/aizenberg/dgellis/fCNN/fcnn/scripts/run_trial.py\
 --config_filename ${CONFIG}\
 --model_filename ${MODEL}\
 --training_log_filename ${LOG}\
 --machine_config_filename ${HCC_CONFIG}
