#!/usr/bin/env python
import glob
import os
import argparse
from multiprocessing import Pool
from functools import partial
import numpy as np
import nibabel as nib
from fcnn.utils.nipy.ggmixture import GGGM
from fcnn.utils.wquantiles.wquantiles import quantile_1D
from fcnn.utils.hcp import get_metric_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namefile", default="/home/aizenberg/dgellis/fCNN/data/labels/ALL-TAVOR_name-file.txt")
    parser.add_argument("--wildcard",
                        default="/work/aizenberg/dgellis/fCNN/predictions/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test/"
                                "*_model_v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_struct14_normalized.midthickness.dscalar.nii")
    parser.add_argument("--target",
                        default="/work/aizenberg/dgellis/HCP/HCP_1200/{subject}/T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/{subject}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii")
    parser.add_argument("--nthreads", default=1)
    return parser.parse_args()


def g2gm_threshold(data, iterations=1000):
    model = GGGM()
    membership = model.estimate(data, niter=iterations)
    lower_threshold = quantile_1D(data, membership[..., 0], 0.5)
    upper_threshold = quantile_1D(data, membership[..., 2], 0.5)
    thresholded_data = np.zeros((2,) + data.shape, int)
    thresholded_data[0][data >= upper_threshold] = 1
    thresholded_data[1][data <= lower_threshold] = 1
    return thresholded_data



def compute_activation_mask(image, subject, metric_names, surface_names):
    data = get_metric_data([image], [metric_names], surface_names=surface_names, subject_id=subject)
    mask_data = list()
    for i, metric_name in enumerate(metric_names):
        thresholded_data = g2gm_threshold(data[..., i])
        mask_data.append(thresholded_data[0])  # just the positive activations for now
    mask_data = np.asarray(mask_data)
    return mask_data

def new_cifti_like(array, cifti):
    return nib.Cifti2Image(dataobj=array.swapaxes(0, 1), header=cifti.header)


def compute_and_save_activation_mask(filename, subject, metric_names, surface_names, output_filename=None):
    if output_filename is None:
        output_filename = filename.replace(".dscalar.nii", ".g2gmactivationmask.dscalar.nii")
    if not os.path.exists(output_filename):
        image = nib.load(filename)
        mask_data = compute_activation_mask(image, subject, metric_names, surface_names)
        new_cifti = new_cifti_like(mask_data, image)
        new_cifti.to_filename(output_filename)


def mp_compute_and_save_activation_mask(args, metric_names, surface_names, output_filename=None):
    filename, subject = args
    print(subject, filename)
    return compute_and_save_activation_mask(filename, subject, metric_names, surface_names,
                                            output_filename=output_filename)
def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def main():
    namespace = parse_args()
    metric_names = read_namefile(namespace.namefile)
    func = partial(mp_compute_and_save_activation_mask,
                   metric_names=metric_names,
                   surface_names=namespace.surface_names)
    args_list = list()
    for fn in glob.glob(namespace.wildcard):
        subject = os.path.basename(fn).split("_")[0]
        args_list.append((fn, subject))
        target_fn = namespace.target.format(subject=subject)
        args_list.append((target_fn, subject))

    pool = Pool(namespace.nthreads)
    pool.map(func, args_list)



if __name__ == "__main__":
    main()
