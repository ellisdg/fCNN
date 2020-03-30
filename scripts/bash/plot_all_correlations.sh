#!/bin/sh
python /home/aizenberg/dgellis/fCNN/scripts/plot_combined_correlations.py\
 /work/aizenberg/dgellis/fCNN/predictions/v4_struct6_unet_MOTOR-TAVOR_2mm_v1_pt/correlations.npy,\
/work/aizenberg/dgellis/fCNN/predictions/v4_struct6_unet_LANGUAGE_2mm_v1_pt/correlations.npy,\
/work/aizenberg/dgellis/fCNN/predictions/v4_struct6_unet_WM_2mm_v1_pt/correlations.npy,\
/work/aizenberg/dgellis/fCNN/predictions/v4_struct6_unet_SOCIAL_2mm_v1_pt/correlations.npy\
 /home/aizenberg/dgellis/fCNN/data/labels/MOTOR-TAVOR_name-file.txt,\
/home/aizenberg/dgellis/fCNN/data/labels/LANGUAGE-TAVOR_name-file.txt,\
/home/aizenberg/dgellis/fCNN/data/labels/WM-TAVOR_name-file.txt,\
/home/aizenberg/dgellis/fCNN/data/labels/SOCIAL-TAVOR_name-file.txt\
 /work/aizenberg/dgellis/fCNN/predictions/figures