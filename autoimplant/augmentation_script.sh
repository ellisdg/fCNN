#!/bin/sh
#SBATCH --time=7-00:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=augment
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load anaconda
conda activate dti
export ANTSPATH=/work/aizenberg/dgellis/tools/ANTs/bin
export PATH=${ANTSPATH}:$PATH

python /home/aizenberg/dgellis/fCNN/autoimplant/autoimplant_augmentation.py --case1 ${1} --case2 ${2} --n_threads 4
