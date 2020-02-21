#!/usr/bin/env bash
conda activate dti
python /home/aizenberg/dgellis/fCNN/scripts/submit_dti_processing.py /work/aizenberg/dgellis/HCP/HCP_1200 /home/aizenberg/dgellis/fCNN/data/subjects_v4.json ${1} /home/aizenberg/dgellis/fCNN/scripts/process_dti_script.py
