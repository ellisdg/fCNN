import os
import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from nilearn.image import new_img_like, resample_to_img
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from unet3d.utils.utils import resample_image_to_spacing


def fetch_dmri_filenames(subject_directory):
    dmri_dir = os.path.join(subject_directory, 'T1w', 'Diffusion')
    data_fn = os.path.join(dmri_dir, 'data.nii.gz')
    bval_fn = os.path.join(dmri_dir, 'bvals')
    bvec_fn = os.path.join(dmri_dir, 'bvecs')
    brainmask_fn = os.path.join(subject_directory, 'T1w', 'brainmask_fs.nii.gz')
    return data_fn, bval_fn, bvec_fn, brainmask_fn


def load_dmri_data(subject_directory):
    data_fn, bval_fn, bvec_fn, brainmask_fn = fetch_dmri_filenames(subject_directory)
    image = nib.load(data_fn)
    bvals, bvecs = read_bvals_bvecs(bval_fn, bvec_fn)
    brainmask = nib.load(brainmask_fn)
    return image, bvals, bvecs, brainmask


def split_dmri_image(image, bvals, bvecs, round_bvals=True, round_decimals=-3, include_b0=True,
                     unique_bvals=(1000, 2000, 3000)):
    output = list()
    if round_bvals:
        _bvals = np.round(bvals, round_decimals)
    else:
        _bvals = np.copy(bvals)
    if unique_bvals is None:
        unique_bvals = np.unique(_bvals)
    for bval in unique_bvals:
        if bval == 0:
            continue
        else:
            output.append(isolate_b_values(image, _bvals, bvecs, target_bval=bval, include_b0=include_b0))
    return output


def isolate_b_values(image, bvals, bvecs, target_bval, include_b0=True):
    bval_mask = bvals == target_bval
    if include_b0:
        bval_mask = np.logical_or(bval_mask, bvals == 0)
    bval_image, bval_bvals, bval_bvecs = isolate_to_mask(image, bvals, bvecs, bval_mask)
    return bval_image, bval_bvals, bval_bvecs


def isolate_to_mask(image, bvals, bvecs, mask):
    masked_image = new_img_like(ref_niimg=image,
                                data=image.get_data()[:, :, :, mask])
    masked_bvals = bvals[mask]
    masked_bvecs = bvecs[mask]
    return masked_image, masked_bvals, masked_bvecs


def random_direction_dti(image, bvals, bvecs, brainmask, target_bval=1000, n_directions=15, n_b0_directions=1,
                         round_bvals=True, round_decimals=-3, spacing=(2, 2, 2), interpolation='linear'):
    if round_bvals:
        _bvals = _bvals = np.round(bvals, round_decimals)
    else:
        _bvals = np.copy(bvals)
    bval_image, bval_bvals, bval_bvecs = isolate_b_values(image, _bvals, bvecs, target_bval=target_bval,
                                                          include_b0=n_b0_directions > 0)
    random_image, random_bvals, random_bvecs = shrink_to_random_directions(bval_image, bval_bvals, bval_bvecs,
                                                                           n_directions=n_directions,
                                                                           n_b0_directions=n_b0_directions)
    if spacing is not None:
        random_image = resample_image_to_spacing(random_image, spacing, interpolation=interpolation)
    tenfit = compute_dti(random_image, random_bvals, random_bvecs, brainmask)
    dti_data = np.concatenate((tenfit.md[..., None], tenfit.color_fa), axis=3)
    dti_image = new_img_like(random_image, dti_data)
    return dti_image


def shrink_to_random_directions(image, bvals, bvecs, n_directions=15, n_b0_directions=1):
    b0_index = np.squeeze(np.where(bvals == 0))
    random_b0_index = np.random.choice(b0_index, n_b0_directions)
    non_b0_index = np.squeeze(np.where(bvals != 0))
    random_non_b0_index = np.random.choice(non_b0_index, n_directions)
    mask = np.zeros(len(bvals), bool)
    mask[random_b0_index] = True
    mask[random_non_b0_index] = True
    random_image, random_bvals, random_bvecs = isolate_to_mask(image, bvals, bvecs, mask)
    return random_image, random_bvals, random_bvecs


def compute_multi_b_value_dti_image(inputs, brainmask):
    dti_data_list = list()
    for image, bvals, bvecs in inputs:
        tenfit = compute_dti(image, bvals, bvecs, brainmask)
        dti_data_list.append(tenfit.md[..., None])
        dti_data_list.append(tenfit.color_fa)
    dti_data = np.concatenate(dti_data_list, axis=3)
    dti_image = new_img_like(ref_niimg=image, data=dti_data)
    return dti_image


def compute_dti(image, bvals, bvecs, brainmask):
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    data = image.get_data()
    brainmask_resampled = resample_to_img(source_img=brainmask,
                                          target_img=image,
                                          interpolation='nearest')
    dwi_masked_data = np.zeros(data.shape)
    mask_index = brainmask_resampled.get_data() > 0
    dwi_masked_data[mask_index] = data[mask_index]
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(dwi_masked_data)
    return tenfit


def process_multi_b_value_dti(subject_directory, output_basename='dti_12.nii.gz', overwrite=False):
    dti_output_filename = os.path.join(subject_directory, 'T1w', 'Diffusion', output_basename)
    if overwrite or not os.path.exists(dti_output_filename):
        image, bvals, bvecs, brainmask = load_dmri_data(subject_directory)
        dti_image = compute_multi_b_value_dti_image(split_dmri_image(image, bvals, bvecs), brainmask)
        dti_image.to_filename(dti_output_filename)


def process_random_direction_dti(subject_directory, output_basename='dti_lowq.nii.gz', overwrite=False):
    dti_output_filename = os.path.join(subject_directory, 'T1w', 'Diffusion', output_basename)
    if overwrite or not os.path.exists(dti_output_filename):
        image, bvals, bvecs, brainmask = load_dmri_data(subject_directory)
        dti_image = random_direction_dti(image, bvals, bvecs, brainmask)
        dti_image.to_filename(dti_output_filename)