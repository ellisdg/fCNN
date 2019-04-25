from fcnn.utils.utils import load_json
import sys
import subprocess


if __name__ == '__main__':
    config = load_json(sys.argv[1])
    script = sys.argv[2]
    model = sys.argv[3]
    for subject_id in config['validation']:
        subprocess.call(['sbatch', script, subject_id, model, config])
