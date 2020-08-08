import subprocess
import glob
import os
import io
import time


def check_queue_length():
    proc = subprocess.Popen(["squeue", "-u", "dgellis"], stdout=subprocess.PIPE)
    return len(list(io.TextIOWrapper(proc.stdout, encoding="utf-8")))


def wait_for_long_queue(sleeping_time=60):
    while check_queue_length() > 1000:
        time.sleep(sleeping_time)


def main():
    skulls = glob.glob("/work/aizenberg/dgellis/MICCAI_Implant_2020/training_set/complete_skull/*.nii.gz")
    cases = [os.path.basename(s) for s in skulls]
    cases2 = list(cases)
    for case in cases:
        cases2.pop(cases.index(case))
        for case2 in cases:
            wait_for_long_queue()
            subprocess.call(["sbatch /home/aizenberg/dgellis/fCNN/autoimplant/augmentation_script.sh", case, case2])


if __name__ == "__main__":
    main()
