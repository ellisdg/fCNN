import sys
import os
import json
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fcnn.predict import make_predictions


if __name__ == '__main__':
    overwrite = False
    config_filename = sys.argv[1]
    print("Config: ", config_filename)
    model_filename = sys.argv[2]
    print("Model: ", model_filename)
    n_subjects = sys.argv[3]
    print("Subjects: ", n_subjects)
    if n_subjects == 'all':
        n_subjects = None
    else:
        n_subjects = int(n_subjects)
    try:
        multiprocessing_config_filename = sys.argv[4]
        with open(multiprocessing_config_filename, 'r') as opened_filename:
            multiprocessing_config = json.load(opened_filename)
        print("MP Config: ", multiprocessing_config_filename)
    except IndexError:
        multiprocessing_config = dict()

    model_basename = os.path.basename(model_filename)
    output_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trials', 'predictions')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    make_predictions(config_filename, model_filename, n_subjects=n_subjects,
                     output_directory=output_directory, overwrite=overwrite,
                     output_replacements=('.func.gii',
                                          '.{}_prediction.func.gii'.format(model_basename.replace(".h5", ''))),
                     **multiprocessing_config)
