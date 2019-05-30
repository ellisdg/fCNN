import os
import json
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from fcnn.utils.utils import normalize_image_data
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
    for subject_id in config['validation'] + config["training"]:
        subject_dir = os.path.join(system_config['directory'], subject_id)
        output_filename = os.path.join(subject_dir, "T1w", "struct6_1.25_normalized.nii.gz")
        print(output_filename)
        if not os.path.exists(output_filename):
            mask_filename = os.path.join(subject_dir, "T1w", "fs_brainmask.nii.gz")
            mask_image = nib.load(mask_filename)
            feature_filenames = [os.path.join(subject_dir, fbn) for fbn in config["feature_basenames"]]
            feature_images = [nib.load(fn) for fn in feature_filenames][::-1]
            image = combine_images(feature_images,
                                   axis=3,
                                   resample_unequal_affines=True,
                                   interpolation="continuous")
            resampled_mask = resample_to_img(mask_image, image, interpolation="nearest")
            crop_affine, crop_shape = crop_img(resampled_mask, return_affine=True, pad=False)
            reordered_affine = reorder_affine(crop_affine, crop_shape)
            image = resample(image, reordered_affine, crop_shape, interpolation="continuous")
            image = image.__class__(normalize_image_data(image.get_data()), image.affine)
            image.to_filename(output_filename)


if __name__ == "__main__":
    main()
