#!/bin/sh
#SBATCH --time=168:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=WB18LS
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu_p100
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=8000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load cuda
module load anaconda
source activate pytorch-1.6

TRIAL=v3_dti4_wb_18_LS_2mm_pt_regularized
CONFIG=/home/aizenberg/dgellis/fCNN/data/${TRIAL}_config.json
HCC_CONFIG=/home/aizenberg/dgellis/fCNN/data/hcc_p100_config.json
MODEL=/work/aizenberg/dgellis/fCNN/model_${TRIAL}.pt
LOG=/work/aizenberg/dgellis/fCNN/log_${TRIAL}.csv

/home/aizenberg/dgellis/.conda/envs/pytorch-1.6/bin/python /home/aizenberg/dgellis/fCNN/fcnn/scripts/run_trial.py ${CONFIG} ${MODEL} ${LOG} ${HCC_CONFIG}
