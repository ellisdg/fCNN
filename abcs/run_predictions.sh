#!/bin/bash
export PYTHONPATH=$HOME/fCNN:$PYTHONPATH

for FOLD in 0 1 2 3 4
do
    echo $FOLD
    python fcnn/scripts/run_unet_inference.py --config_filename "/home/aizenberg/dgellis/fCNN/abcs/config/abcs_config_fold${FOLD}.json" --model_filename "/work/aizenberg/dgellis/MICCAI_ABCs_2020/models/cross_validation/model_abcs_fold${FOLD}.h5" --output_directory "/work/aizenberg/dgellis/MICCAI_ABCs_2020/predictions/test1/abcs_fold${FOLD}_raw" --group test1 --output_template "UNMCNeuroSurg_18092020_{subject}.nii.gz" --replace "augmented_training_data" "test1"
    python fcnn/scripts/run_unet_inference.py --config_filename "/home/aizenberg/dgellis/fCNN/abcs/config/abcs_config_fold${FOLD}.json" --model_filename "/work/aizenberg/dgellis/MICCAI_ABCs_2020/models/cross_validation/model_abcs_fold${FOLD}_best.h5" --output_directory "/work/aizenberg/dgellis/MICCAI_ABCs_2020/predictions/test1/abcs_fold${FOLD}_best_raw" --group test1 --output_template "UNMCNeuroSurg_18092020_{subject}.nii.gz" --replace "augmented_training_data" "test1"
done
