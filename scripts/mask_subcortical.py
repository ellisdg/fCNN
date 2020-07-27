import os
import nibabel as nib
import numpy as np
import argparse
from functools import partial
from multiprocessing import Pool
from fcnn.utils.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', default="/home/aizenberg/dgellis/fCNN/data/subjects_v4-retest.json")
    parser.add_argument('--hcp_dir', default="/work/aizenberg/dgellis/HCP/HCP_1200")
    parser.add_argument('--group', default="all")
    parser.add_argument('--input_filename', default="T1w_acpc_dc_restore_brain.nii.gz")
    parser.add_argument('--output_filename', default="T1w_acpc_dc_restore_subcortical.nii.gz")
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--cortex', action="store_true", default=False)
    parser.add_argument('--overwrite', action="store_true", default=False)
    return vars(parser.parse_args())


def mask_t1_image(subject, hcp_dir, input_filename, output_filename, overwrite=False, cortex=False):
    output_filename = os.path.join(hcp_dir, subject, "T1w", output_filename)
    t1_filename = os.path.join(hcp_dir, subject, "T1w", input_filename)
    if (overwrite or not os.path.exists(output_filename)) and os.path.exists(t1_filename):
        print(output_filename)
        t1 = nib.load(t1_filename)
        aparc_filename = os.path.join(hcp_dir, subject, "T1w", "aparc+aseg.nii.gz")
        aparc = nib.load(aparc_filename)
        atlas = aparc.get_data()
        exclude = [46, 47, 7, 8, 15, 16]
        excluded_mask = np.in1d(atlas.flat, exclude)
        if cortex:
            also_excluded = atlas.flat > 1000
        else:
            also_excluded = np.logical_and(atlas.flat < 1000, atlas.flat > 0)
        also_excluded[excluded_mask] = False
        t1_data = t1.get_fdata()
        t1_data.flat[also_excluded == False] = 0.
        wm_t1 = t1.__class__(dataobj=t1_data, header=t1.header, affine=t1.affine)
        wm_t1.to_filename(output_filename)


def main():
    args = parse_args()
    subjects_dict = load_json(args["subjects"])
    if args["group"] == "all":
        subjects = list()
        for key in subjects_dict:
            subjects.extend(subjects_dict[key])
    else:
        subjects = subjects_dict[args["group"]]
    func = partial(mask_t1_image, hcp_dir=args["hcp_dir"], input_filename=args["input_filename"],
                   output_filename=args["output_filename"], overwrite=args["overwrite"])
    if args["nthreads"] > 1:
        pool = Pool(args["nthreads"])
        pool.map(func, subjects)
    else:
        for subject in subjects:
            func(subject)


if __name__ == "__main__":
    main()
