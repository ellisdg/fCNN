import sys
import os
import json
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fcnn.predict import make_predictions
import subprocess


if __name__ == '__main__':
    overwrite = False
    config_filename = sys.argv[1]
    print("Config: ", config_filename)
    model_filename = sys.argv[2]
    print("Model: ", model_filename)
    subject = sys.argv[3]
    print("Subject: ", subject)
    multiprocessing_config_filename = sys.argv[4]
    with open(multiprocessing_config_filename, 'r') as opened_filename:
        multiprocessing_config = json.load(opened_filename)
    print("MP Config: ", multiprocessing_config_filename)
    output_directory = os.path.abspath(sys.argv[5])
    print("Output Directory:", output_directory)
    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        print(nvidia_smi_output)
        if 'P100' in nvidia_smi_output:
            batch_size = 50
        elif 'K40' in nvidia_smi_output:
            batch_size = 5
        elif 'K20' in nvidia_smi_output:
            batch_size = 1
        else:
            batch_size = 1
    except subprocess.CalledProcessError or FileNotFoundError:
        batch_size = 1
    model_basename = os.path.basename(model_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    make_predictions(config_filename, model_filename, n_subjects=None, single_subject=subject, batch_size=batch_size,
                     output_directory=output_directory, overwrite=overwrite,
                     output_replacements=('.func.gii',
                                          '.{}_prediction.func.gii'.format(model_basename.replace(".h5", ''))),
                     **multiprocessing_config)
