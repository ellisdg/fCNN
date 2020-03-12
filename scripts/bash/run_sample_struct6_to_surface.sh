#!/bin/bash
module load awscli
conda activate fcnn-cpu-1.7
export PYTHONPATH=/home/aizenberg/dgellis/fCNN:$PYTHONPATH
SurfaceName=${1}
Method=enclosing
python scripts/sample_volume_to_surface.py /work/aizenberg/dgellis/HCP/HCP_1200 data/subjects_v4.json "T1w/fsaverage_LR32k/{subject}.{hemi}.$SurfaceName.32k_fs_LR.gii" ${Method} T1w/struct6_nomralized.nii.gz data/labels/STRUCT6_name-file.txt $SurfaceName
