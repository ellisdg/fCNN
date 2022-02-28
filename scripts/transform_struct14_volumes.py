from fcnn.utils.utils import load_json
import subprocess
import argparse
from multiprocessing import Pool
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="test")
    parser.add_argument("--subjects_file", default="/home/aizenberg/dgellis/fCNN/data/subjects_v4.json")
    parser.add_argument("--nthreads", default=1, type=int)
    parser.add_argument("--hcp_dir", default="/work/aizenberg/dgellis/HCP/HCP_1200")
    return parser.parse_args()


def call_script(subject,
                script="/home/aizenberg/dgellis/fCNN/scripts/bash/transform_struct14_volumes.sh",
                hcp_dir="/work/aizenberg/dgellis/HCP/HCP_1200"):
    cmd = ["bash", script, subject, hcp_dir]
    print(" ".join(cmd))
    subprocess.call(cmd)


def main():
    namespace = parse_args()
    subjects = load_json(namespace.subjects_file)
    func = partial(call_script, hcp_dir=namespace.hcp_dir)
    with Pool(namespace.nthreads) as pool:
        pool.map(func, subjects[namespace.group])


if __name__ == "__main__":
    main()
