#!/bin/sh
#SBATCH --time=168:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=predict
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load anaconda
source activate fcnn-cpu-1.7

HCC_CONFIG=/home/aizenberg/dgellis/fCNN/data/hcc_predict_config_cpu.json
OUTPUT_DIR=/work/aizenberg/dgellis/fCNN/predictions
SUBJECT_ID=${1}
MODEL=${2}
CONFIG=${3}

export PYTHONPATH=/home/aizenberg/dgellis/3DUnetCNN:$PYTHONPATH
export KERAS_BACKEND=tensorflow

/home/aizenberg/dgellis/.conda/envs/fcnn-1.12/bin/python /home/aizenberg/dgellis/fCNN/trials/run_prediction.py ${CONFIG} ${MODEL} ${SUBJECT_ID} ${HCC_CONFIG} ${OUTPUT_DIR}
