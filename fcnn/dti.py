import os
import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from nilearn.image import new_img_like, resample_to_img
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti


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
            bval_mask = _bvals == bval
            if include_b0:
                bval_mask = np.logical_or(bval_mask, _bvals == 0)
            bval_image = new_img_like(ref_niimg=image,
                                      data=image.get_data()[:, :, :, bval_mask])
            bval_bvals = bvals[bval_mask]
            bval_bvecs = bvecs[bval_mask]
            output.append((bval_image, bval_bvals, bval_bvecs))
    return output


def compute_dti_image(inputs, brainmask):
    dti_data_list = list()
    for image, bvals, bvecs in inputs:
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
        dti_data_list.append(tenfit.md[..., None])
        dti_data_list.append(tenfit.color_fa)
    dti_data = np.concatenate(dti_data_list, axis=3)
    dti_image = new_img_like(ref_niimg=image, data=dti_data)
    return dti_image


def process_dti(subject_directory, output_basename='dti_12.nii.gz'):
    image, bvals, bvecs, brainmask = load_dmri_data(subject_directory)
    dti_image = compute_dti_image(split_dmri_image(image, bvals, bvecs), brainmask)
    dti_image.to_filename(os.path.join(subject_directory, 'T1w', 'Diffusion', output_basename))
