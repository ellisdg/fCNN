import subprocess
import argparse
import os
import glob
from functools import partial
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--hcp_dir", default="/work/aizenberg/dgellis/HCP/HCP_1200")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--wb_command", default="/work/aizenberg/dgellis/tools/workbench/bin_rh_linux64/wb_command")
    return parser.parse_args()


def transform_to_mni_space(in_file, command, xfm, reference, method, out_file, fnirt_file):
    cmd = [command, "-volume-warpfield-resample", in_file,
           xfm, reference, method, out_file,
           "-fnirt", fnirt_file]
    print(" ".join(cmd))
    subprocess.call(cmd)


def proc_and_transform(in_file, command, hcp_dir, method="CUBIC"):
    subject_id = os.path.basename(in_file).split("_")[0]
    xfm = os.path.join(hcp_dir, subject_id, "MNINonLinear/xfms/acpc_dc2standard.nii.gz")
    reference = os.path.join(hcp_dir, subject_id, "MNINonLinear/T1w_restore.2.nii.gz")
    fnirt_file = os.path.join(hcp_dir, subject_id, "T1w/T1w_acpc_dc_restore_brain.nii.gz")
    out_file = os.path.join(os.path.dirname(in_file), "MNI", os.path.basename(in_file))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    transform_to_mni_space(in_file, command, xfm, reference, method, out_file, fnirt_file)


def main():
    namespace = parse_args()
    func = partial(proc_and_transform, command=namespace.wb_command, hcp_dir=namespace.hcp_dir)
    filenames = glob.glob(os.path.join(namespace.prediction_dir, "*.nii.gz"))

    with Pool(namespace.nthreads) as pool:
        pool.map(func, filenames)


if __name__ == "__main__":
    main()
