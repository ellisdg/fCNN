#!/usr/bin/env bash
TRIAL=${1}
MODEL=/work/aizenberg/dgellis/fCNN/model_${TRIAL}.pt
CONF=/home/aizenberg/dgellis/fCNN/data/${TRIAL}.csv
MCONF=/home/aizenberg/dgellis/fCNN/data/hcc_p100_config.json
ODIR=/work/aizenberg/dgellis/fCNN/predictions
BIAS=/work/aizenberg/dgellis/HCP/HCP_1200/validation_tfMRI_LANGUAGE_level2_hp200_s2.dscalar.nii
SID=198855
/home/aizenberg/dgellis/.conda/envs/fcnn-1.12/bin/python /home/aizenberg/dgellis/fCNN/fcnn/scripts/make_whole_brain_predictions.py ${CONF} ${MODEL} ${MCONF} ${ODIR} ${BIAS} ${SID}
