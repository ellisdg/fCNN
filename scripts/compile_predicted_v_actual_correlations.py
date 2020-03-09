import sys
import glob
import os
from fcnn.utils.utils import load_json, update_progress
from fcnn.utils.hcp import get_metric_data, extract_cifti_scalar_data
from scipy.stats import pearsonr, ks_2samp
import nibabel as nib
from multiprocessing import Pool
from functools import partial
import numpy as np


def normalize_correlation_matrix_by_axis(matrix, new_max, new_min, axis=0):
    matrix = (matrix - matrix.min(axis=axis)) / (matrix.max(axis=axis) - matrix.min(axis=axis))
    matrix = ((new_max - new_min) * matrix) + new_min
    return matrix


def normalize_correlation_matrix(matrix, new_max, new_min, axes=(0, 1)):
    for axis in axes:
        matrix = normalize_correlation_matrix_by_axis(matrix, new_max, new_min, axis=axis)
    return matrix


def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def compute_correlation_row(predicted_fn, target_fns, metric_names, structure_names, pool_size=None):
    predicted_image = nib.load(predicted_fn)
    predicted_data = get_metric_data([predicted_image], [metric_names], structure_names, None)
    func = partial(compute_correlation, predicted_data=predicted_data, metric_names=metric_names,
                   structure_names=structure_names)
    if pool_size is not None:
        pool = Pool(pool_size)
        return pool.map(func, target_fns)
    else:
        return [func(fn) for fn in target_fns]


def compute_correlation(target_fn, predicted_data, metric_names, structure_names):
    target_image = nib.load(target_fn)
    target_data = get_metric_data([target_image], [metric_names], structure_names, None)
    task_row = list()
    for i, task_name in enumerate(metric_names):
        task_row.append(pearsonr(predicted_data[..., i].flatten(), target_data[..., i].flatten()))
    return task_row


def main():
    output_file = sys.argv[1]
    config_filename = "/home/neuro-user/PycharmProjects/fCNN/data/v4_struct6_unet_MOTOR-TAVOR_2mm_v1_pt_config.json"
    hcp_dir = "/media/crane/HCP/HCP_1200"
    config = load_json(config_filename)
    target_basename = config["target_basenames"]
    prediction_dir = "/home/neuro-user/PycharmProjects/fCNN/trials/predictions/v4_struct6_unet_MOTOR-TAVOR_2mm_v1_pt"
    all_prediction_images = glob.glob(os.path.join(prediction_dir, "*.dscalar.nii"))
    target_images = list()
    structure_names = ["CortexLeft", "CortexRight"]
    # hemispheres = ["L", "R"]
    # surface_template = "T1w/fsaverage_LR32k/{subject}.{hemi}.{surf}.32k_fs_LR.surf.gii"
    # all_surfaces = list()
    prediction_images = list()
    surf_name = "midthickness"
    metric_filename = "/home/neuro-user/PycharmProjects/fCNN/data/labels/MOTOR-TAVOR_name-file.txt"
    metric_names = read_namefile(metric_filename)
    pool_size = None

    for p_image_fn in all_prediction_images:
        if "target" not in p_image_fn:
            sid = os.path.basename(p_image_fn).split("_")[0]
            target_fn = os.path.join(hcp_dir, sid, target_basename.format(sid)).replace(".nii.gz",
                                                                                        ".{}.dscalar.nii").format(
                surf_name)
            target_images.append(target_fn)
            prediction_images.append(p_image_fn)
            # all_surfaces.append([os.path.join(hcp_dir, sid, surface_template.format(subject=sid,
            #                                                                         hemi=hemi,
            #                                                                         surf=surf_name)) for hemi in
            #                      hemispheres])
    correlations = list()
    for i, p_image_fn in enumerate(prediction_images):
        update_progress(i/len(prediction_images), message=os.path.basename(p_image_fn).split("_")[0])
        correlations.append(compute_correlation_row(p_image_fn, target_images, metric_names, structure_names,
                                                    pool_size=pool_size))
    update_progress(1)
    np.save(output_file, correlations)


if __name__ == "__main__":
    main()
