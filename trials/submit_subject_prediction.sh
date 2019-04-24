#!/bin/sh
#SBATCH --time=168:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=predict
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load cuda
module load anaconda
source activate fcnn-1.12

CONFIG=/home/aizenberg/dgellis/fCNN/data/trial_34_LS_LM_config.json
HCC_CONFIG=/home/aizenberg/dgellis/fCNN/data/hcc_predict_config.json
MODEL=/work/aizenberg/dgellis/fCNN/model_trial_lowq_32_LS_LM.h5
OUTPUT_DIR=/work/aizenberg/dgellis/fCNN/predictions

export PYTHONPATH=/home/aizenberg/dgellis/3DUnetCNN:$PYTHONPATH
export KERAS_BACKEND=tensorflow

/home/aizenberg/dgellis/.conda/envs/fcnn-1.12/bin/python /home/aizenberg/dgellis/fCNN/trials/run_prediction.py ${CONFIG} ${MODEL} ${1} ${HCC_CONFIG} ${OUTPUT_DIR}
