import sys
import os
import argparse
import subprocess
from multiprocessing import Pool
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fcnn.dti import process_dti, process_multi_b_value_dti
from fcnn.utils.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_filename')
    parser.add_argument('--group')
    parser.add_argument('--hcp_dir')
    parser.add_argument('--multi_b_value', action='store_true', default=False)
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--submit', action='store_true', default=False)
    parser.add_argument('--subject_dir')
    return vars(parser.parse_args())


def submit(subject_dirs, python="/home/aizenberg/dgellis/.conda/envs/dti/bin/python", nthreads=1, mem_per_cpu=8000,
           flags="", delete_script=True):

    for subject_dir in subject_dirs:
        slurm_script_filename = os.path.join(".", "DTI_{}.slurm".format(os.path.basename(subject_dir)))
        sbatch = """
        #!/bin/bash
        #SBATCH --time=7-00:00:00          # Run time in hh:mm:ss
        #SBATCH --mem_per_cpu-per-cpu={}       # Maximum memory required per CPU (in megabytes)
        #SBATCH --ntasks-per-node={}
        #SBATCH --job-name=DTI_{}
        #SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.DTI_%J.err
        #SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.DTI_%J.err
        conda activate dti
        export PYTHONPATH=/home/aizenberg/dgellis/fCNN:$PYTHONPATH
        {} {} --subject_dir {}{}
        """.format(
            nthreads, mem_per_cpu, os.path.basename(subject_dir), python, __file__, subject_dir, flags)
        with open(slurm_script_filename) as temp:
            temp.write(sbatch)
        cmd = ['sbatch', slurm_script_filename]
        subprocess.call(cmd)
        if delete_script:
            os.remove(slurm_script_filename)


def main():
    args = parse_args()
    if args['subject_dir'] is not None:
        subject_dirs = [args['subject_dir']]
    else:
        group = args['group']
        subjects = load_json(args["subjects_filename"])[group]
        hcp_dir = args["hcp_dir"]
        subject_dirs = [os.path.join(hcp_dir, subject) for subject in subjects]
    if args['submit']:
        flags = ""
        if args['multi_b_value']:
            flags += " --multi_b_value"
        submit(subject_dirs, flags=flags, nthreads=args['nthreads'])
    else:
        if args['multi_b_value']:
            process_func = process_multi_b_value_dti
        else:
            process_func = process_dti
        if args['nthreads'] > 1:
            pool = Pool(args['nthreads'])
            pool.map(process_func, subject_dirs)
        else:
            for subject_dir in subject_dirs:
                process_func(subject_dir)


if __name__ == '__main__':
    main()
