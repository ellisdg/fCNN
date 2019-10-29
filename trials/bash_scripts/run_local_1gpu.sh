#!/usr/bin/env bash

export PYTHONPATH=/home/neuro-user/PycharmProjects/fCNN/3DUNetCNN:/home/neuro-user/PycharmProjects/fCNN:$PYTHONPATH

TRIAL=${1}
CONFIG=/home/neuro-user/PycharmProjects/fCNN/data/${TRIAL}_config.json
HCC_CONFIG=/home/neuro-user/PycharmProjects/fCNN/data/local_config_1gpu.json
MODEL=/home/neuro-user/PycharmProjects/fCNN/trials/models/model_${TRIAL}.h5
LOG=/home/neuro-user/PycharmProjects/fCNN/trials/training_logs/log_${TRIAL}.csv

python /home/neuro-user/PycharmProjects/fCNN/fcnn/scripts/run_trial.py ${CONFIG} ${MODEL} ${LOG} ${HCC_CONFIG}
