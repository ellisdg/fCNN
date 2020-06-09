import sys
import os
import argparse
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
    return vars(parser.parse_args())


def main():
    args = parse_args()
    group = args['group']
    subjects = load_json(args["subjects_filename"])[group]
    hcp_dir = args["hcp_dir"]
    subject_dirs = [os.path.join(hcp_dir, subject) for subject in subjects]
    pool = Pool(args['nthreads'])
    if args['multi_b_value']:
        process_func = process_multi_b_value_dti
    else:
        process_func = process_dti
    pool.map(process_func, subject_dirs)


if __name__ == '__main__':
    main()
