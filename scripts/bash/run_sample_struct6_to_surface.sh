#!/bin/bash
module load awscli
module load anaconda
conda activate dti
export PYTHONPATH=/home/aizenberg/dgellis/fCNN:$PYTHONPATH
export PATH=/work/aizenberg/dgellis/tools/workbench/bin_rh_linux64:$PATH
SurfaceName=${1}
Method=enclosing
${HOME}/.conda/envs/dti/bin/python scripts/sample_volume_to_surface.py /work/aizenberg/dgellis/HCP/HCP_1200 data/subjects_v4.json "T1w/fsaverage_LR32k/{subject}.{hemi}.$SurfaceName.32k_fs_LR.surf.gii" ${Method} T1w/struct6_normalized.nii.gz data/labels/STRUCT6_name-file.txt "MNINonLinear/fsaverage_LR32k/{subject}.{hemi}.atlasroi.32k_fs_LR.shape.gii" $SurfaceName
