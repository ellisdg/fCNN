import subprocess
import glob
import os
import io
import time
import copy


def check_queue_length():
    proc = subprocess.Popen(["squeue", "-u", "dgellis"], stdout=subprocess.PIPE)
    return len(list(io.TextIOWrapper(proc.stdout, encoding="utf-8")))


def wait_for_long_queue(sleeping_time=60):
    while check_queue_length() > 1000:
        time.sleep(sleeping_time)


def main():
    skulls = glob.glob("/work/aizenberg/dgellis/MICCAI_Implant_2020/training_set/complete_skull/*.nii.gz")
    cases1 = sorted([os.path.basename(s) for s in skulls])
    cases2 = copy.copy(cases1)
    for i, case1 in enumerate(cases1):
        for case2 in cases2[(i+1):]:
            wait_for_long_queue()
            subprocess.call(["sbatch /home/aizenberg/dgellis/fCNN/autoimplant/augmentation_script.sh", case1, case2])


if __name__ == "__main__":
    main()
