#!/bin/sh
#SBATCH --time=48:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=correlations
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out
TASK=${1}
module load anaconda
conda activate dti
fCNNDir=/home/aizenberg/dgellis/fCNN
export PYTHONPATH=$fCNNDir:/home/aizenberg/dgellis/3DUnetCNN:PYTHONPATH
PDIR=/work/aizenberg/dgellis/fCNN/predictions/v4_struct6_unet_${TASK}_2mm_v1_pt
python $fCNNDir/scripts/compile_predicted_v_actual_correlations.py ${PDIR}/correlations.npy $fCNNDir/data/v4_struct6_unet_${TASK}-TAVOR_2mm_v1_pt_config.json ${PDIR} $fCNNDir/data/labels/${TASK}-TAVOR_name-file.txt
