#!/bin/bash
module load awscli
module load anaconda
conda activate dti
export PYTHONPATH=/home/aizenberg/dgellis/fCNN:$PYTHONPATH
SurfaceName=${1}
Method=enclosing
${HOME}/.conda/envs/dti/bin/python scripts/sample_volume_to_surface.py /work/aizenberg/dgellis/HCP/HCP_1200 data/subjects_v4.json "T1w/fsaverage_LR32k/{subject}.{hemi}.$SurfaceName.32k_fs_LR.gii" ${Method} T1w/struct6_nomralized.nii.gz data/labels/STRUCT6_name-file.txt $SurfaceName
