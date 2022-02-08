import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import new_img_like
from .compute_activation_masks import g2gm_threshold
import argparse
from multiprocessing import Pool
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wildcard",
                        default="/work/aizenberg/dgellis/fCNN/predictions/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test/MNI/"
                                "*_model_v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_struct14_normalized.nii.gz")
    parser.add_argument("--target",
                        default="/work/aizenberg/dgellis/HCP/HCP_1200/{subject}/MNINonLinear/Results/tfMRI_ALL/"
                                "tfMRI_ALL_hp200_s2_level2.feat/{subject}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.nii.gz")
    parser.add_argument("--nthreads", default=1, type=int)
    parser.add_argument("--overwrite", default=False, action="store_true")
    return parser.parse_args()


def threshold_4d_volume(data, threshold_func=g2gm_threshold, **threshold_kwargs):
    data = np.copy(data)
    for volume_index in range(data.shape[-1]):
        _data = data[..., volume_index]
        _shape = _data.shape
        _thresholded = threshold_func(_data.ravel(), **threshold_kwargs)[0].reshape(_shape)  # just positive activations
        data[..., volume_index] = _thresholded
    return data


def threshold_4d_nifti_volume(filename, output_filename=None, overwite=False):
    if output_filename is None:
        output_filename = filename.replace(".nii", "_g2gmactivationmask.nii")
    if overwite or not os.path.exists(output_filename):
        image = nib.load(filename)
        data = threshold_4d_volume(image.get_fdata())
        output_image = new_img_like(image, data)
        return output_image.to_filename(output_filename)


def main():
    namespace = parse_args()
    filenames = glob.glob(namespace.wildcard)
    target_filenames = list()

    for i, filename in enumerate(filenames):
        subject = os.path.basename(filename).split("_")[0]
        target_fn = namespace.target.format(subject=subject)
        target_filenames.append(target_fn)

    func = partial(threshold_4d_nifti_volume, overwrite=namespace.overwrite)

    with Pool(namespace.nthreads) as pool:
        pool.map(func, filenames + target_filenames)


if __name__ == "__main__":
    main()
