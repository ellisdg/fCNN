#!/bin/sh
#SBATCH --time=48:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=vol_convert
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out
module load awscli
module load anaconda
conda activate dti
export PYTHONPATH=/home/aizenberg/dgellis/fCNN:$PYTHONPATH
export PATH=/work/aizenberg/dgellis/tools/workbench/bin_rh_linux64:$PATH
Basename=TAVOR
fCNNDir=/home/aizenberg/dgellis/fCNN
HCPDir=/work/aizenberg/dgellis/HCP/HCP_1200
HCPRetestDir=/work/aizenberg/dgellis/HCP/HCP_Retest
Method="ribbon-constrained"
SurfaceName=midthickness
SubjectsConfig=${fCNNDir}/data/subjects_v4-retest.json
for Dir in ${HCPDir} ${HCPRetestDir}
do
  # convert the cifti volumes to nifti volumes
  for Task in "MOTOR" "LANGUAGE" "WM" "RELATIONAL" "EMOTION" "SOCIAL" "GAMBLING"
  do
    ${HOME}/.conda/envs/dti/bin/python ${fCNNDir}/scripts/convert_cifti_volumes.py data/v4_struct6_unet_${Task}_2mm_v1_pt_config.json ${Dir} ${Basename} ${SubjectsConfig}
  done
  # combine the nifti volumes
  ${HOME}/.conda/envs/dti/bin/python ${fCNNDir}/scripts/convert_cifti_volumes ${SubjectsConfig} ${Dir}
  # sample the nifti volumes to the surface
  ${HOME}/.conda/envs/dti/bin/python ${fCNNDir}/scripts/sample_volume_to_surface.py ${Dir} ${SubjectsConfig} "T1w/fsaverage_LR32k/{subject}.{hemi}.$SurfaceName.32k_fs_LR.surf.gii" ${Method} "T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/{subject}_tfMRI_ALL_level2_zstat_hp200_s2_${Basename}.nii.gz" $fCNNDir/data/labels/ALL-TAVOR_name-file.txt "MNINonLinear/fsaverage_LR32k/{subject}.{hemi}.atlasroi.32k_fs_LR.shape.gii" $SurfaceName
done

