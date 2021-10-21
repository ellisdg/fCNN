#!/bin/sh
#SBATCH --time=7-00:00:00
#SBATCH --job-name=correlations
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4000
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load anaconda
conda activate fcnn-1.12
export PYTHONPATH=$HOME/fCNN:$PYTHONPATH
target=${1}
partial=${2}
surface=${3}

python $HOME/fCNN/scripts/compile_predicted_v_actual_overall_correlations.py --output_dir $WORK/fCNN/predictions/v4_${target}_unet_ALL-TAVOR_2mm_v2_pt_test --output_filename $WORK/fCNN/predictions/v4_${target}_unet_ALL-TAVOR_2mm_v2_pt_test/overall_partial_cov_${partial}_${surface}_correlations.npy --config_filename $HOME/fCNN/data/v4_${target}_unet_ALL-TAVOR_2mm_v2_pt_config.json --nthreads 16 --verbose --surface_name "${surface}" --covariate_dir $WORK/fCNN/predictions/v4_${partial}_unet_ALL-TAVOR_2mm_v2_pt_test
