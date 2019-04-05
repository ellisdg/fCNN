#!/bin/sh
#SBATCH --time=168:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=dti_proc
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load anaconda
source activate dti

export PYTHONPATH=/home/aizenberg/dgellis/3DUnetCNN:$PYTHONPATH

OUTPUT="$(which python)"
echo "${OUTPUT}"
/home/aizenberg/dgellis/.conda/envs/dti/bin/python /home/aizenberg/dgellis/fCNN/scripts/process_low_quality_dti.py ${1} ${2}
