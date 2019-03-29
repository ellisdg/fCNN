import sys
import os
import json
import math
import subprocess


def main():
    hcp_dir = os.path.abspath(sys.argv[1])
    config_filename = os.path.abspath((sys.argv[2]))
    subjects_per_process = int(sys.argv[3])
    script_filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'run_dti_processing.sh')
    with open(config_filename) as opened_file:
        config = json.load(opened_file)

    subject_ids = config['training'] + config['validation']
    n_procs = int(math.ceil(float(len(subject_ids)/float(subjects_per_process))))
    for i in range(n_procs):
        arg_subjects = ",".join(subject_ids[i*subjects_per_process:(i+1)*subjects_per_process])
        cmd = ['sbatch', script_filename, arg_subjects, hcp_dir]
        print(" ".join(cmd))
        subprocess.call(cmd)
        break


if __name__ == '__main__':
    main()
