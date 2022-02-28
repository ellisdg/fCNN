#!/bin/sh
#SBATCH --time=7-00:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=V100_2GPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu_v100
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out

module load cuda
module load anaconda
ENV="pytorch-1.8"
conda activate ${ENV}

export PYTHONPATH=/home/aizenberg/dgellis/fCNN:/home/aizenberg/dgellis/3DUnetCNN:${PYTHONPATH}

TRIAL=${1}
CONFIG=/home/aizenberg/dgellis/fCNN/data/${TRIAL}_config.json
HCC_CONFIG=/home/aizenberg/dgellis/fCNN/data/hcc_v100_2gpu_32gb_config.json
MODEL=/work/aizenberg/dgellis/fCNN/model_${TRIAL}.h5
LOG=/work/aizenberg/dgellis/fCNN/log_${TRIAL}.csv

/home/aizenberg/dgellis/.conda/envs/${ENV}/bin/python /home/aizenberg/dgellis/fCNN/fcnn/scripts/run_trial.py\
 --config_filename ${CONFIG}\
 --model_filename ${MODEL}\
 --training_log_filename ${LOG}\
 --machine_config_filename ${HCC_CONFIG}

OUTPUT_DIRECTORY=/work/aizenberg/dgellis/fCNN/predictions/${TRIAL}

/home/aizenberg/dgellis/.conda/envs/${ENV}/bin/python /home/aizenberg/dgellis/fCNN/fcnn/scripts/run_unet_inference.py\
 --config_filename ${CONFIG}\
 --model_filename ${MODEL}\
 --machine_config_filename ${HCC_CONFIG}\
 --output_directory ${OUTPUT_DIRECTORY}

/home/aizenberg/dgellis/.conda/envs/${ENV}/bin/python fCNN/scripts/sample_volume_to_surface.py\
 --volume_template "${OUTPUT_DIRECTORY}/{subject}_model_${TRIAL}_struct14_normalized.nii.gz"\
 --config $HOME/fCNN/data/subjects_v4-retest.json

/home/aizenberg/dgellis/.conda/envs/${ENV}/bin/python fCNN/scripts/compile_predicted_v_actual_overall_correlations.py\
 --output_filename ${OUTPUT_DIRECTORY}/overall_correlations.npy\
 --output_dir ${OUTPUT_DIRECTORY}\
 --config_filename ${CONFIG}\
 --nthreads 40 --verbose
