import glob
import os
import subprocess


def main():
    ct = "/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data/{}_ct.nii.gz"
    t1 = "/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data/{}_t1.nii.gz"
    t2 = "/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data/{}_t2.nii.gz"
    subjects = [os.path.basename(f).split("_")[0] for f in
                glob.glob("/work/aizenberg/dgellis/MICCAI_ABCs_2020/*_ct.nii.gz")]
    for i in range(len(subjects)):
        s1 = subjects[i]
        for s2 in subjects[(i + 1):]:
            bn = "{}_to_{}".format(s1, s2)
            cmd = ['antsRegistrationSyNQuick.sh',
                             '-f',
                             ct.format(s1),
                             '-f',
                             t1.format(s1),
                             '-f',
                             t2.format(s1),
                             '-m',
                             ct.format(s2),
                             '-m',
                             t1.format(s2),
                             '-m',
                             t2.format(s2),
                             '-n',
                             '32',
                             '-d',
                             '3',
                             '-o',
                             bn]
            print(" ".join(cmd))
            subprocess.call(cmd)


if __name__ == "__main__":
    main()
