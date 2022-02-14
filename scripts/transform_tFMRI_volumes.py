from fcnn.utils.utils import load_json
import subprocess
import argparse
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="test")
    parser.add_argument("--subjects_file", default="/home/aizenberg/dgellis/fCNN/data/subjects_v4.json")
    parser.add_argument("--nthreads", type=int, default=1)
    return parser.parse_args()


def transform_subject(subject):
    cmd = ["bash", "/home/aizenberg/dgellis/fCNN/scripts/bash/transform_tfMRI_volumes.sh", subject]
    print(" ".join(cmd))
    subprocess.call(["bash", "/home/aizenberg/dgellis/fCNN/scripts/bash/transform_tfMRI_volumes.sh", subject])


def main():
    namespace = parse_args()
    subjects = load_json(namespace.subjects_file)[namespace.group]
    with Pool(namespace.nthreads) as pool:
        pool.map(transform_subject, subjects)


if __name__ == "__main__":
    main()
