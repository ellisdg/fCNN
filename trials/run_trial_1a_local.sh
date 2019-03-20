#!/usr/bin/env bash

conda activate mappingdb

TRIAL=trial_1a
CONFIG=/home/neuro-user/PycharmProjects/fCNN/data/trial1_local_config.json
HCC_CONFIG=/home/neuro-user/PycharmProjects/fCNN/data/local_config.json
MODEL=/home/neuro-user/PycharmProjects/fCNN/trials/${TRIAL}/model_${TRIAL}.h5
LOG=/home/neuro-user/PycharmProjects/fCNN/trials/${TRIAL}/log_${TRIAL}.csv

python /home/neuro-user/PycharmProjects/fCNN/trials/run_trial.py ${CONFIG} ${MODEL} ${LOG} ${HCC_CONFIG}
