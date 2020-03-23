#!/bin/bash
module load awscli
module load anaconda
conda activate dti
export PYTHONPATH=/home/aizenberg/dgellis/fCNN:$PYTHONPATH
export PATH=/work/aizenberg/dgellis/tools/workbench/bin_rh_linux64:$PATH
Basename=TAVOR
fCNNDir=/home/aizenberg/dgellis/fCNN
HCPDir=/work/aizenberg/dgellis/HCP/HCP_1200
Method="ribbon-constrained"
SurfaceName=midthickness
for TASK in SOCIAL RELATIONAL GAMBLING EMOTION
do
   ${HOME}/.conda/envs/dti/bin/python scripts/convert_cifti_volumes.py data/v4_struct6_unet_${TASK}_2mm_v1_pt_config.json ${HCPDir} ${Basename}
   ${HOME}/.conda/envs/dti/bin/python scripts/sample_volume_to_surface.py /work/aizenberg/dgellis/HCP/HCP_1200 $fCNNDir/data/subjects_v4.json "T1w/fsaverage_LR32k/{subject}.{hemi}.$SurfaceName.32k_fs_LR.surf.gii" ${Method} "T1w/Results/tfMRI_${TASK}/tfMRI_${TASK}_hp200_s2_level2.feat/{subject}_tfMRI_${TASK}_level2_zstat_hp200_s2.nii.gz" $fCNNDir/data/labels/${TASK}-TAVOR_name-file.txt "MNINonLinear/fsaverage_LR32k/{subject}.{hemi}.atlasroi.32k_fs_LR.shape.gii" $SurfaceName
done