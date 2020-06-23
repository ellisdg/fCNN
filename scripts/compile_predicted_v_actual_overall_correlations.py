import glob
import sys
import os
import argparse
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


def compute_correlation_row(predicted_fn, target_fns, metric_names, structure_names, verbose=False, level="overall"):
    if verbose:
        print(predicted_fn)
    predicted_image = nib.load(predicted_fn)
    predicted_data = get_metric_data_for_metric_names(predicted_image, metric_names, structure_names, None)
    row = list()
    for fn in target_fns:
        row.append(compute_correlation(target_fn=fn, predicted_data=predicted_data, metric_names=metric_names,
                                       structure_names=structure_names, level=level))
    return row


def get_metric_data_for_metric_names(target_image, metric_names, structure_names, subject=None):
    try:
        return get_metric_data([target_image], [metric_names], structure_names, subject)
    except ValueError:
        _metric_names = [metric_name.split(" ")[-1] for metric_name in metric_names]
        return get_metric_data([target_image], [_metric_names], structure_names, subject)


def compute_correlation(target_fn, predicted_data, metric_names, structure_names, level="overall"):
    target_image = nib.load(target_fn)
    target_data = get_metric_data_for_metric_names(target_image, metric_names, structure_names, None)
    if level == "overall":
        return pearsonr(predicted_data.flatten(), target_data.flatten())
    elif level == "task":
        task_row = list()
        for i, task_name in enumerate(metric_names):
            task_row.append(pearsonr(predicted_data[..., i].flatten(), target_data[..., i].flatten()))
        return task_row
    elif level == "domain":
        domains = np.asarray([task.split(" ")[0] for task in metric_names])
        domain_row = list()
        processed_domains = list()
        for domain in domains:
            if domain not in processed_domains:
                domain_mask = domains == domain
                domain_row.append(pearsonr(predicted_data[..., domain_mask].flatten(),
                                           target_data[..., domain_mask].flatten()))
                processed_domains.append(domain)
    else:
        raise NotImplementedError("Level='{}'".format(level))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_filename', required=True)
    parser.add_argument('--config_filename', required=True)
    parser.add_argument('--prediction_dir', required=True)
    parser.add_argument('--hcp_dir', default="/work/aizenberg/dgellis/HCP/HCP_1200")
    parser.add_argument('--task_names', default="/home/aizenberg/dgellis/fCNN/data/labels/ALL-TAVOR_name-file.txt")
    parser.add_argument('--surface_name', default="midthickness")
    parser.add_argument('--level', default="overall", choices=['overall', 'domain', 'task'])
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--structures', nargs=2, default=["CortexLeft", "CortexRight"])
    parser.add_argument('--verbose', action="store_true", default=False)
    parser.add_argument('--submit', action="store_true", default=False)
    return vars(parser.parse_args())


def main():
    args = parse_args()
    if args["submit"]:
        from .process_dti_script import submit_slurm_script
        flags = list()
        for arg in sys.argv[1:]:
            if arg != "--submit":
                flags.append(arg)
        return submit_slurm_script(nthreads=args["nthreads"], mem_per_cpu=4000, python_file=__file__,
                                   job_name=os.path.basename(args["output_filename"].split(".")[0]),
                                   slurm_script_filename=args["output_filename"].replace(".npy", ".slurm"),
                                   flags=" ".join(flags))
    output_file = args["output_filename"]
    config_filename = args["config_filename"]
    hcp_dir = args["hcp_dir"]
    config = load_json(config_filename)
    target_basename = config["target_basenames"]
    prediction_dir = args["prediction_dir"]
    metric_filename = args["task_names"]
    surf_name = args["surface_name"]
    all_prediction_images = glob.glob(os.path.join(prediction_dir, "*.{}.dscalar.nii".format(surf_name)))
    target_images = list()
    structure_names = args["structures"]
    prediction_images = list()
    metric_names = read_namefile(metric_filename)
    pool_size = args["nthreads"]
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

    correlations = list()
    if pool_size > 1:
        func = partial(compute_correlation_row, target_fns=target_images, metric_names=metric_names,
                       structure_names=structure_names, verbose=args["verbose"], level=args["level"])
        pool = Pool(pool_size)
        correlations = pool.map(func, prediction_images)
    else:
        for i, p_image_fn in enumerate(prediction_images):
            update_progress(i/len(prediction_images), message=os.path.basename(p_image_fn).split("_")[0])
            correlations.append(compute_correlation_row(p_image_fn, target_images, metric_names, structure_names,
                                                        level=args["level"]))
        update_progress(1)
    np.save(output_file, correlations)
    np.save(output_file.replace(".npy", "_subjects.npy"), subjects)


if __name__ == "__main__":
    main()
