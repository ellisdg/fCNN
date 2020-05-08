import sys
import glob
import os
from functools import partial
from multiprocessing import Pool
from fcnn.utils.utils import load_json, update_progress
from fcnn.utils.hcp import get_metric_data, extract_cifti_scalar_map_names
from scipy.stats import pearsonr
import nibabel as nib
import numpy as np


def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def compute_correlation_row(predicted_fn, target_fns, metric_names, structure_names, pool_size=None, verbose=False,
                            fix_metric_names=True):
    if verbose:
        print(predicted_fn)
    predicted_image = nib.load(predicted_fn)
    predicted_data = get_metric_data([predicted_image], [metric_names], structure_names, None)
    row = list()
    for fn in target_fns:
        row.append(compute_correlation(target_fn=fn, predicted_data=predicted_data, metric_names=metric_names,
                                       structure_names=structure_names, pool_size=pool_size,
                                       fix_metric_names=fix_metric_names))
    return row


def compute_correlation(target_fn, predicted_data, metric_names, structure_names, pool_size=None,
                        fix_metric_names=True):
    target_image = nib.load(target_fn)
    if fix_metric_names:
        target_metric_names = [metric_name.split(" ")[-1] for metric_name in metric_names]
    else:
        target_metric_names = metric_names
    target_data = get_metric_data([target_image], [target_metric_names], structure_names, None)
    task_row = list()
    for i, task_name in enumerate(metric_names):
        task_row.append(pearsonr(predicted_data[..., i].flatten(), target_data[..., i].flatten()))
    return task_row


def main():
    output_file = sys.argv[1]
    config_filename = sys.argv[2]
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    config = load_json(config_filename)
    target_basename = config["target_basenames"]
    prediction_dir = sys.argv[3]
    metric_filename = sys.argv[4]
    surf_name = sys.argv[5]
    all_prediction_images = glob.glob(os.path.join(prediction_dir, "*.{}.dscalar.nii".format(surf_name)))
    target_images = list()
    structure_names = ["CortexLeft", "CortexRight"]
    prediction_images = list()
    metric_names = read_namefile(metric_filename)
    pool_size = 32
    subjects = list()
    for p_image_fn in all_prediction_images:
        if "target" not in p_image_fn:
            sid = os.path.basename(p_image_fn).split("_")[0]
            subjects.append(sid)
            target_fn = os.path.join(hcp_dir, sid, target_basename.format(sid)).replace(".nii.gz",
                                                                                        ".{}.dscalar.nii").format(
                surf_name)
            target_images.append(target_fn)
            prediction_images.append(p_image_fn)

    target = nib.load(target_images[0])
    if np.all(np.in1d(metric_names, extract_cifti_scalar_map_names(target))):
        fix_metric_names = False
    else:
        fix_metric_names = True

    correlations = list()
    if pool_size is not None:
        func = partial(compute_correlation_row, pool_size=None, target_fns=target_images, metric_names=metric_names,
                       structure_names=structure_names, verbose=True, fix_metric_names=fix_metric_names)
        pool = Pool(pool_size)
        correlations = pool.map(func, prediction_images)
    else:
        for i, p_image_fn in enumerate(prediction_images):
            update_progress(i/len(prediction_images), message=os.path.basename(p_image_fn).split("_")[0])
            correlations.append(compute_correlation_row(p_image_fn, target_images, metric_names, structure_names,
                                                        pool_size=pool_size))
        update_progress(1)
    np.save(output_file, correlations)
    np.save(output_file.replace(".npy", "_subjects.npy"), subjects)


if __name__ == "__main__":
    main()
