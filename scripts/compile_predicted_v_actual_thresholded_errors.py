import sys
import glob
import os
from functools import partial
from multiprocessing import Pool
from fcnn.utils.utils import load_json, update_progress
from fcnn.utils.hcp import get_metric_data
from fcnn.utils.nipy.ggmixture import GGGM
from fcnn.utils.wquantiles.wquantiles import quantile_1D
import pandas as pd
import nibabel as nib
import numpy as np


def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def compute_errors(predicted_fn, target_fn, group_average_data_thresholded, metric_names, structure_names, verbose=False):
    if verbose:
        print(predicted_fn)
    predicted_image = nib.load(predicted_fn)
    predicted_data = get_metric_data([predicted_image], [metric_names], structure_names, None)
    target_image = nib.load(target_fn)
    target_data = get_metric_data([target_image], [metric_names], structure_names, None)
    prediction_errors = list()
    group_average_errors = list()
    for index, metric_name in enumerate(metric_names):
        predicted_data_thresholded = g2gm_threshold(predicted_data[index])
        target_data_thresholded = g2gm_threshold(target_data[index])
        predicted_dice = compute_dice(predicted_data_thresholded, target_data_thresholded)
        prediction_errors.append(predicted_dice)
        group_average_errors.append(compute_dice(target_data_thresholded,
                                                 group_average_data_thresholded[index]))
    return prediction_errors + group_average_errors


def compute_mae(data1, data2):
    return np.mean(np.abs(data1 - data2))


def compute_mse(data1, data2):
    return np.mean(np.square(data1-data2))


def g2gm_threshold(data, iterations=1000):
    print(data.shape)
    model = GGGM()
    membership = model.estimate(data, niter=iterations)
    lower_threshold = quantile_1D(data, membership[..., 0], 0.5)
    print(lower_threshold)
    upper_threshold = quantile_1D(data, membership[..., 2], 0.5)
    print(upper_threshold)
    thresholded_data = np.zeros((2,) + data.shape, np.int)
    thresholded_data[0][data >= upper_threshold] = 1
    thresholded_data[1][data <= lower_threshold] = 1
    print(np.sum(thresholded_data[0]), np.sum(thresholded_data[1]))
    return thresholded_data


def compute_dice(x, y):
    return 2. * (x * y).sum()/(x.sum() + y.sum())


def threshold_data_for_all_metrics(data, metric_names):
    thresholded_data = list()
    for index, metric_name in enumerate(metric_names):
        print("Group Average:", metric_name)
        thresholded_data.append(g2gm_threshold(data[index]))
    return np.asarray(thresholded_data)


def main():
    output_file = sys.argv[1]
    config_filename = sys.argv[2]
    prediction_dir = sys.argv[3]
    name_file = sys.argv[4]
    group_average_filename = sys.argv[5]

    target_images = list()
    prediction_images = list()

    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    config = load_json(config_filename)
    target_basename = config["target_basenames"]
    surf_name = "midthickness"
    all_prediction_images = glob.glob(os.path.join(prediction_dir, "*.{}.dscalar.nii".format(surf_name)))
    structure_names = ["CortexLeft", "CortexRight"]
    metric_names = read_namefile(name_file)
    corrected_metric_names = [m.split(" ")[-1] for m in metric_names]

    group_average_image = nib.load(group_average_filename)
    group_average_data = get_metric_data([group_average_image], metric_names=[corrected_metric_names],
                                         surface_names=structure_names, subject_id=None)
    group_average_thresholded_data = threshold_data_for_all_metrics(group_average_data,
                                                                    metric_names=corrected_metric_names)

    pool_size = None
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
    errors = list()
    if pool_size is not None:
        func = partial(compute_errors, metric_names=corrected_metric_names, structure_names=structure_names,
                       verbose=True)
        pool = Pool(pool_size)
        errors = pool.map(func, prediction_images)
    else:
        for i, (p_image_fn, t_image_fn) in enumerate(zip(prediction_images, target_images)):
            update_progress(i/len(prediction_images), message=os.path.basename(p_image_fn).split("_")[0])
            errors.append(compute_errors(predicted_fn=p_image_fn, target_fn=t_image_fn,
                                         metric_names=corrected_metric_names,
                                         structure_names=structure_names,
                                         group_average_data_thresholded=group_average_thresholded_data))
        update_progress(1)
    df = pd.DataFrame(errors, index=subjects, columns=metric_names + ["Group Average " + m for m in metric_names])
    df.to_csv(output_file)


if __name__ == "__main__":
    main()
