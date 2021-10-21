#!/bin/sh
#SBATCH --time=7-00:00:00
#SBATCH --job-name=correlations
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=16000
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load anaconda
conda activate nib
export PATH=$WORK/tools/workbench/bin_rh_linux64:$PATH

python $HOME/fCNN/scripts/transform_pred_volumes.py --prediction_dir "${1}" --nthreads 4
