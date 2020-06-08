#!/bin/bash
SUBFN=/home/aizenberg/dgellis/fCNN/data/subjects_v4.json
TEMPLATE="/work/aizenberg/dgellis/HCP/HCP_1200/{subject}/T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/{subject}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii"
OUT_TEMPLATE="/work/aizenberg/dgellis/fCNN/v4_{group}_{operation}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii"
python /home/aizenberg/dgellis/fCNN/scripts/compute_group_mean.py --group test --subjects_filename ${SUBFN}\
 --template ${TEMPLATE} --output_filename ${OUT_TEMPLATE}