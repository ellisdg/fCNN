import os
import json
from multiprocessing import Pool
from functools import partial
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img


def load_json(filename):
    with open(filename, 'r') as opened_file:
        return json.load(opened_file)


def combine_images(images, axis=0, resample_unequal_affines=False, interpolation="linear"):
    base_image = images[0]
    data = list()
    max_dim = len(base_image.shape)
    for image in images:
        try:
            np.testing.assert_array_equal(image.affine, base_image.affine)
        except AssertionError as error:
            if resample_unequal_affines:
                image = resample_to_img(image, base_image, interpolation=interpolation)
            else:
                raise error
        image_data = image.get_data()
        dim = len(image.shape)
        if dim < max_dim:
            image_data = np.expand_dims(image_data, axis=axis)
        elif dim > max_dim:
            max_dim = max(max_dim, dim)
            data = [np.expand_dims(x, axis=axis) for x in data]
        data.append(image_data)
    if len(data[0].shape) > 3:
        array = np.concatenate(data, axis=axis)
    else:
        array = np.stack(data, axis=axis)
    return base_image.__class__(array, base_image.affine)


def main():
    config = load_json(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "subjects_v4.json"))
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    feature_basenames = ["T1w/T1w_acpc_dc_restore_brain.nii.gz",
                         "T1w/T2w_acpc_dc_restore_brain.nii.gz"]
    overwrite = False
    subject_ids = config['validation'] + config["training"] + config["test"]
    func = partial(write_image, hcp_dir=hcp_dir, feature_basenames=feature_basenames, overwrite=overwrite)
    with Pool(16) as pool:
        pool.map(func, subject_ids)


def write_image(subject_id, hcp_dir, feature_basenames, output_basename="T1T2w_acpc_dc_restore_brain.nii.gz",
                overwrite=True):
    subject_dir = os.path.join(hcp_dir, subject_id)
    output_filename = os.path.join(subject_dir, "T1w", output_basename)
    print(output_filename)
    if overwrite or not os.path.exists(output_filename):
        feature_filenames = [os.path.join(subject_dir, fbn) for fbn in feature_basenames]
        feature_images = [nib.load(fn) for fn in feature_filenames][::-1]
        image = combine_images(feature_images,
                               axis=3,
                               resample_unequal_affines=False,
                               interpolation="continuous")
        image.to_filename(output_filename)


if __name__ == "__main__":
    main()
