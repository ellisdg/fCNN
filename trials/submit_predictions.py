from fcnn.utils.utils import load_json
import sys
import subprocess


if __name__ == '__main__':
    config = load_json(sys.argv[1])
    script = sys.argv[2]
    for subject_id in config['validation']:
        subprocess.call(['bash', script, subject_id])
