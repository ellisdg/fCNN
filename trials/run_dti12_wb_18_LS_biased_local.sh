#!/usr/bin/env bash

TRIAL=dti12_wb_18_LS_biased
CONFIG=/home/neuro-user/PycharmProjects/fCNN/data/${TRIAL}_config.json
HCC_CONFIG=/home/neuro-user/PycharmProjects/fCNN/data/local_config.json
MODEL=/home/neuro-user/PycharmProjects/fCNN/trials/models/model_${TRIAL}.h5
LOG=/home/neuro-user/PycharmProjects/fCNN/trials/training_logs/log_${TRIAL}.csv

python /home/neuro-user/PycharmProjects/fCNN/trials/run_trial.py ${CONFIG} ${MODEL} ${LOG} ${HCC_CONFIG}
