import os
import glob


def main():
    for training_job_fn in glob.glob("/work/aizenberg/dgellis/MICCAI_BraTS2020/models/job.brats_fold*_*.err"):
        _, fold, jobid, = os.path.basename(training_job_fn).split("_")



if __name__ == "__main__":
    main()
