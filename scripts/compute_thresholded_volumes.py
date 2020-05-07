import sys
import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import new_img_like
from fcnn.utils.nipy.ggmixture import GGGM
from fcnn.utils.wquantiles.wquantiles import quantile_1D
from fcnn.utils.utils import update_progress


def g2gm_threshold(data, iterations=1000, lower_quantile=0.5, upper_quantile=0.5, background_value=0):
    """
    Guassian 2 gamma thresholding.

    :param data: input statistical map data.
    :param iterations: numer of iterations for EM algorithm.
    :param lower_quantile: quantile of the negative gamma to threshold below.
    :param upper_quantile: quantile of the positive gamma to threshold above.
    :param background_value: background value to for the data outside the thresholds (default = 0).
    :return: data with values below the upper threshold and above the lower threshold set to zero.
    """
    model = GGGM()
    membership = model.estimate(data, niter=iterations)
    lower_threshold = quantile_1D(data, membership[..., 0], lower_quantile)
    upper_threshold = quantile_1D(data, membership[..., 2], upper_quantile)
    thresholded_data = np.copy(data)[np.logical_and(data < upper_threshold, data > lower_threshold)] = background_value
    return thresholded_data


def threshold_4d_volume(data, threshold_func=g2gm_threshold, **threshold_kwargs):
    data = np.copy(data)
    for volume_index in range(data.shape[-1]):
        _data = data[..., volume_index]
        _shape = _data.shape
        _thresholded = threshold_func(_data.ravel(), **threshold_kwargs).reshape(_shape)
        data[..., volume_index] = _thresholded
    return data


def threshold_4d_nifti_volume(filename, output_filename):
    image = nib.load(filename)
    data = threshold_4d_volume(image.get_fdata())
    output_image = new_img_like(image, data)
    return output_image.to_filename(output_filename)


def main():
    wildcard = sys.argv[1]
    filenames = glob.glob(wildcard)
    overwrite = True
    for i, filename in enumerate(filenames):
        update_progress(i/len(filenames))
        output_filename = filename.replace(".nii", "_thresholded.nii")
        if overwrite or not os.path.exists(output_filename):
            threshold_4d_nifti_volume(filename, output_filename)
    update_progress(1)


if __name__ == "__main__":
    main()
