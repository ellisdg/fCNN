from fcnn.utils.utils import load_json
import sys
import subprocess
import os

if __name__ == '__main__':
    config_filename = os.path.abspath(sys.argv[1])
    config = load_json(config_filename)
    script = os.path.abspath(sys.argv[2])
    model_filename = os.path.abspath(sys.argv[3])
    for subject_id in config['validation']:
        subprocess.call(['sbatch', script, subject_id, model_filename, config_filename])
