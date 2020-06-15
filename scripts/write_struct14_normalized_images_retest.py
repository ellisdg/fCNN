import os
import json
from multiprocessing import Pool
from functools import partial
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from fcnn.utils.utils import zero_one_window, zero_floor_normalize_image_data
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
    config = load_json(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "subjects_v4-retest.json"))
    subject_ids = config['retest']
    feature_basenames = ["T1w/T1w_acpc_dc_restore_brain.nii.gz",
                         "T1w/T2w_acpc_dc_restore_brain.nii.gz",
                         "T1w/Diffusion/dti_12.nii.gz"]
    channels_to_normalize = [True, True,
                             True, False, False, False,
                             True, False, False, False,
                             True, False, False, False]
    for directory in ["/work/aizenberg/dgellis/HCP/HCP_1200", "/work/aizenberg/dgellis/HCP/HCP_Retest"]:
        func = partial(write_struct6_image,
                       hcp_dir=directory,
                       feature_basenames=feature_basenames,
                       channels_to_normalize=channels_to_normalize,
                       overwrite=True,
                       output_channels=(((2, 14), "T1w/Diffusion/dti_12_normalized.nii.gz"),),
                       normalization_kwargs={"floor_percentile": 25,
                                             "ceiling_percentile": 99.9})
        with Pool(8) as pool:
            pool.map(func, subject_ids)


def write_struct6_image(subject_id, hcp_dir, feature_basenames, channels_to_normalize, overwrite=False,
                        normalize_func=zero_one_window, output_channels=None, crop=False,
                        normalization_kwargs=None):
    if normalization_kwargs is None:
        normalization_kwargs = dict()
    subject_dir = os.path.join(hcp_dir, str(subject_id))
    output_filename = os.path.join(subject_dir, "T1w", "struct14_normalized.nii.gz")
    print(output_filename)
    if overwrite or not os.path.exists(output_filename):
        feature_filenames = [os.path.join(subject_dir, fbn) for fbn in feature_basenames]
        try:
            feature_images = [nib.load(fn) for fn in feature_filenames]
        except FileNotFoundError:
            return
        image = combine_images(feature_images,
                               axis=3,
                               resample_unequal_affines=True,
                               interpolation="linear")
        if crop:
            mask_filename = os.path.join(subject_dir, "T1w", "brainmask_fs.nii.gz")
            mask_image = nib.load(mask_filename)
            resampled_mask = resample_to_img(mask_image, image, interpolation="nearest")
            crop_affine, crop_shape = crop_img(resampled_mask, return_affine=True, pad=False)
            reordered_affine = reorder_affine(crop_affine, crop_shape)
            image = resample(image, reordered_affine, crop_shape, interpolation="linear")
        image_data = image.get_fdata()
        image_data_list = list()
        for channel, normalize in zip(range(image.shape[3]), channels_to_normalize):
            data = image_data[..., channel]
            if normalize:
                data = normalize_func(data, **normalization_kwargs)
            image_data_list.append(data)
        image_data = np.moveaxis(np.asarray(image_data_list), 0, 3)
        image = image.__class__(image_data, image.affine)
        image.to_filename(output_filename)
        if output_channels is not None:
            for (start, stop), basename in output_channels:
                output_channel_filename = os.path.join(subject_dir, basename)
                print(output_channel_filename)
                if overwrite or os.path.exists(output_channel_filename):
                    image.__class__(np.squeeze(image_data[..., start:stop]),
                                    image.affine).to_filename(output_channel_filename)


if __name__ == "__main__":
    main()
