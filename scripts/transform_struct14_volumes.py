from fcnn.utils.utils import load_json
import subprocess
import argparse
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="test")
    parser.add_argument("--subjects_file", default="/home/aizenberg/dgellis/fCNN/data/subjects_v4.json")
    parser.add_argument("--nthreads", default=1, type=int)
    return parser.parse_args()


def call_script(subject, script="/home/aizenberg/dgellis/fCNN/scripts/bash/transform_struct14_volumes.sh"):
    cmd = ["bash", "/home/aizenberg/dgellis/fCNN/scripts/bash/transform_tfMRI_volumes.sh", subject]
    print(" ".join(cmd))
    subprocess.call(cmd)


def main():
    namespace = parse_args()
    subjects = load_json(namespace.subjects_file)
    with Pool(namespace.nthreads) as pool:
        pool.map(call_script, subjects[namespace.group])


if __name__ == "__main__":
    main()
