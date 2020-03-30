#!/bin/sh
#SBATCH --time=7-00:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=correlations
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out
export PYTHONPATH=/home/aizenberg/dgellis/fCNN:/home/aizenberg/dgellis/3DUnetCNN:PYTHONPATH
TASK=${1}
# bash /home/aizenberg/dgellis/fCNN/scripts/bash/convert_tfMRI_prediction_dir.sh ${TASK}
bash /home/aizenberg/dgellis/fCNN/scripts/bash/compile_correlations.sh ${TASK}
