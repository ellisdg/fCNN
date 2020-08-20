#!/usr/bin/env bash
TRIAL=${1}
SYSTEM=${2}
MODEL=/work/aizenberg/dgellis/fCNN/model_${TRIAL}.pt
CONF=/home/aizenberg/dgellis/fCNN/data/${TRIAL}_config.json
MCONF=/home/aizenberg/dgellis/fCNN/data/hcc_${SYSTEM}_config.json
ODIR=/work/aizenberg/dgellis/fCNN/predictions
BIAS=/work/aizenberg/dgellis/HCP/HCP_1200/v3_training_tfMRI_LANGUAGE_level2_hp200_s2.dscalar.nii
SID=100206
/home/aizenberg/dgellis/.conda/envs/pytorch-1.5/bin/python /home/aizenberg/dgellis/fCNN/fcnn/scripts/make_whole_brain_predictions.py ${CONF} ${MODEL} ${MCONF} ${ODIR} ${BIAS} ${SID}
