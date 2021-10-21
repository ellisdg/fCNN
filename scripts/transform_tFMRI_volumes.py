from fcnn.utils.utils import load_json
import subprocess


def main():
    subjects = load_json("/home/aizenberg/dgellis/fCNN/data/subjects_v4.json")
    for subject in subjects["test"]:
        subprocess.call(["bash", "/home/aizenberg/dgellis/fCNN/scripts/bash/transform_tfMRI_volumes.sh", subject])


if __name__ == "__main__":
    main()
