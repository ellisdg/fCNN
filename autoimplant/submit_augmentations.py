import subprocess
import glob
import os


def main():
    skulls = glob.glob("/work/aizenberg/dgellis/MICCAI_Implant_2020/training_set/complete_skull/*.nii.gz")
    cases = [os.path.basename(s) for s in skulls]
    cases2 = list(cases)
    for case in cases:
        cases2.pop(cases.index(case))
        for case2 in cases:
            subprocess.call(["sbatch /home/aizenberg/dgellis/fCNN/autoimplant/augmentation_script", case, case2])


if __name__ == "__main__":
    main()
