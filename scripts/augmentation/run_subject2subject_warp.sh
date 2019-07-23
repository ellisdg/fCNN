#!/usr/bin/env bash
HCP=/work/aizenberg/dgellis/HCP/HCP_1200
SUB=/home/aizenberg/dgellis/fCNN/data/subjects_v3.json
OUT=/work/aizenberg/dgellis/HCP/xfms
SCRIPT=/home/aizenberg/dgellis/fCNN/scripts/augmentation/combine_hcp_warps.sh
REL=T1w/Diffusion/dti.nii.gz

export PYTHONPATH=/home/aizenberg/dgellis/fCNN:${PYTHONPATH}

python /home/aizenberg/dgellis/fCNN/fcnn/scripts/augmentation/subject2subject_warps.py
