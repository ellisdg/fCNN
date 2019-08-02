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
source activate fcnn-1.12

TRIAL=dti4_wb_18_LS_2mm_biased_augmented
CONFIG=/home/aizenberg/dgellis/fCNN/data/${TRIAL}_config.json
HCC_CONFIG=/home/aizenberg/dgellis/fCNN/data/hcc_p100_config.json
MODEL=/work/aizenberg/dgellis/fCNN/model_${TRIAL}.h5
LOG=/work/aizenberg/dgellis/fCNN/log_${TRIAL}.csv

export PYTHONPATH=/home/aizenberg/dgellis/3DUnetCNN:$PYTHONPATH
export KERAS_BACKEND=tensorflow

OLD_MODEL=/work/aizenberg/dgellis/fCNN/model_dti4_wb_18_LS.h5
BIAS=/work/aizenberg/dgellis/HCP/HCP_1200/training_tfMRI_LANGUAGE_level2_hp200_s2.dscalar.nii

/home/aizenberg/dgellis/.conda/envs/fcnn-1.12/bin/python /home/aizenberg/dgellis/fCNN/scripts/set_model_bias.py ${OLD_MODEL} ${CONFIG} ${BIAS} ${MODEL}
/home/aizenberg/dgellis/.conda/envs/fcnn-1.12/bin/python /home/aizenberg/dgellis/fCNN/trials/run_trial.py ${CONFIG} ${MODEL} ${LOG} ${HCC_CONFIG} ${BIAS}
