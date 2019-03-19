import sys
import os
import json
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fcnn.train import run_training


if __name__ == '__main__':
    config_filename = sys.argv[0]
    model_filename = sys.argv[1]
    training_log_filename = sys.argv[2]
    try:
        multiprocessing_config_filename = sys.argv[3]
        with open(multiprocessing_config_filename, 'r') as opened_filename:
            multiprocessing_config = json.load(opened_filename)
    except IndexError:
        multiprocessing_config = dict()

    run_training(config_filename, model_filename, training_log_filename, **multiprocessing_config)
