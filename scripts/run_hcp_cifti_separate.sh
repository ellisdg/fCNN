#!/bin/sh
#SBATCH --time=24:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=4024       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/registration_jobs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/registration_jobs/job.%J.out

subject_id=$1
hcp_dir=/work/aizenberg/dgellis/HCP/HCP_1200
results_dir=${hcp_dir}/${subject_id}/MNINonLinear/Results
smoothing_level=4

for task_name in MOTOR LANGUAGE
do
    task_results_dir=${results_dir}/tfMRI_${task_name}/tfMRI_${task_name}_hp200_s${smoothing_level}_level2_MSMAll.feat/
    cifti_basename=${task_results_dir}/${subject_id}_tfMRI_${task_name}_level2_hp200_s${smoothing_level}_MSMAll
    cifti_filename=${cifti_basename}.dscalar.nii
    left_output=${cifti_basename}.L.func.gii
    right_output=${cifti_basename}.R.func.gii
    /work/aizenberg/dgellis/tools/workbench/bin_rh_linux64/wb_command -cifti-separate ${cifti_filename} COLUMN -metric CORTEX_LEFT ${left_output} -metric CORTEX_RIGHT ${right_output}
done
