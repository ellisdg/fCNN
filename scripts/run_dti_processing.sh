#!/bin/sh
#SBATCH --time=168:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=dti_proc
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load anaconda
conda activate dti

OUTPUT="$(which python)"
echo "${OUTPUT}"
export PYTHONPATH=/home/aizenberg/dgellis/fCNN:/home/aizenberg/dgellis/3DUnetCNN:${PYTHONPATH}
/home/aizenberg/dgellis/.conda/envs/dti/bin/python ${3} ${1} ${2}
