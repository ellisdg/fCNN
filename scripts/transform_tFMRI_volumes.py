from fcnn.utils.utils import load_json
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="test")
    parser.add_argument("--subjects_file", default="/home/aizenberg/dgellis/fCNN/data/subjects_v4.json")
    return parser.parse_args()


def main():
    namespace = parse_args()
    subjects = load_json(namespace.subjects_file)
    for subject in subjects[namespace.group]:
        subprocess.call(["bash", "/home/aizenberg/dgellis/fCNN/scripts/bash/transform_tfMRI_volumes.sh", subject])


if __name__ == "__main__":
    main()
