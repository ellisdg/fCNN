from fcnn.utils.utils import load_json
import subprocess
import argparse
from multiprocessing import Pool
import os
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="test")
    parser.add_argument("--subjects_file", default="/home/aizenberg/dgellis/fCNN/data/subjects_v4.json")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--check_output_template",
                        default="/work/aizenberg/dgellis/HCP/HCP_1200/{subject}/MNINonLinear/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/{subject}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.nii.gz")
    return parser.parse_args()


def transform_subject(subject, output_template, overwrite=False):
    output_filename = output_template.format(subject=subject)
    if overwrite or not os.path.exists(output_filename):
        cmd = ["bash", "/home/aizenberg/dgellis/fCNN/scripts/bash/transform_tfMRI_volumes.sh", subject]
        print(" ".join(cmd))
        subprocess.call(["bash", "/home/aizenberg/dgellis/fCNN/scripts/bash/transform_tfMRI_volumes.sh", subject])
    else:
        print("Already exists:", output_filename)


def main():
    namespace = parse_args()
    subjects = load_json(namespace.subjects_file)[namespace.group]
    func = partial(transform_subject, overwrite=namespace.overwrite,
                   output_template=namespace.check_output_template)
    with Pool(namespace.nthreads) as pool:
        pool.map(func, subjects)


if __name__ == "__main__":
    main()
