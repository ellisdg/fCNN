#!/bin/sh
#SBATCH --time=168:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=trial_5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu_k40
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load cuda
module load anaconda
source activate pytorch-1.6

TRIAL=trial_5
CONFIG=/home/aizenberg/dgellis/fCNN/data/${TRIAL}_config.json
HCC_CONFIG=/home/aizenberg/dgellis/fCNN/data/hcc_k40_config.json
MODEL=/work/aizenberg/dgellis/fCNN/model_${TRIAL}.h5
LOG=/work/aizenberg/dgellis/fCNN/log_${TRIAL}.csv

export PYTHONPATH=/home/aizenberg/dgellis/3DUnetCNN:$PYTHONPATH

python /home/aizenberg/dgellis/fCNN/trials/run_trial.py ${CONFIG} ${MODEL} ${LOG} ${HCC_CONFIG}
