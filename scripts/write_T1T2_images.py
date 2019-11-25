import os
import json
from multiprocessing import Pool
from functools import partial
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from fcnn.utils.utils import zero_mean_normalize_image_data
from unet3d.utils.nilearn_custom_utils.nilearn_utils import crop_img, reorder_affine
from unet3d.utils.utils import resample


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
    config = load_json(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "data",
                                    "t1t2dti4_wb_18_LS_config.json"))
    system_config = load_json(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           "data",
                                           "hcc_p100_config.json"))
    subject_ids = config['validation'] + config["training"]
    func = partial(write_image, system_config=system_config, config=config, overwrite=False)
    with Pool(16) as pool:
        pool.map(func, subject_ids)


def write_image(subject_id, system_config, config, overwrite=False):
    subject_dir = os.path.join(system_config['directory'], subject_id)
    output_filename = os.path.join(subject_dir, "T1w", "T1T2w_acpc_dc_restore_brain.nii.gz")
    print(output_filename)
    if overwrite or not os.path.exists(output_filename):
        feature_filenames = [os.path.join(subject_dir, fbn) for fbn in config["feature_basenames"]]
        feature_images = [nib.load(fn) for fn in feature_filenames][::-1]
        image = combine_images(feature_images,
                               axis=3,
                               resample_unequal_affines=False,
                               interpolation="continuous")
        image.to_filename(output_filename)


if __name__ == "__main__":
    main()
