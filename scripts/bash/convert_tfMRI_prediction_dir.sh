#!/bin/bash
module load awscli
module load anaconda
conda activate dti
export PYTHONPATH=/home/aizenberg/dgellis/fCNN:$PYTHONPATH
export PATH=/work/aizenberg/dgellis/tools/workbench/bin_rh_linux64:$PATH
SurfaceName=midthickness
fCNNDir=/home/aizenberg/dgellis/fCNN
Method="ribbon-constrained"
TASK=${1}
PredDir=/work/aizenberg/dgellis/fCNN/predictions/v4_struct6_unet_${Task}_2mm_v1_pt
${HOME}/.conda/envs/dti/bin/python $fCNNDir/sample_volume_to_surface.py /work/aizenberg/dgellis/HCP/HCP_1200 $fCNNDir/data/subjects_v4.json "T1w/fsaverage_LR32k/{subject}.{hemi}.$SurfaceName.32k_fs_LR.surf.gii" ${Method} "${PredDir}/{subject}_model_v4_struct6_unet_${TASK}_2mm_v1_pt_struct6_normalized.nii.gz" $fCNNDir/data/labels/${TASK}-TAVOR_name-file.txt "MNINonLinear/fsaverage_LR32k/{subject}.{hemi}.atlasroi.32k_fs_LR.shape.gii" $SurfaceName
