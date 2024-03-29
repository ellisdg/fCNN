import os
from functools import partial
from multiprocessing import Pool
from fcnn.utils.utils import load_json, update_progress
from fcnn.utils.hcp import get_metric_data
from scipy.stats import pearsonr
import nibabel as nib
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_filename', required=True)
    parser.add_argument('--test_dir', default="/work/aizenberg/dgellis/HCP/HCP_1200")
    parser.add_argument('--retest_dir', default="/work/aizenberg/dgellis/HCP/HCP_Retest")
    parser.add_argument('--subjects_filename', default="/home/aizenberg/dgellis/fCNN/data/subjects_v4-retest.json")
    parser.add_argument('--group', default="retest")
    # parser.add_argument('--surface_name', default="midthickness")
    parser.add_argument('--template',
                        default="T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/"
                                "{subject}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii")
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--metric_names', default="/home/aizenberg/dgellis/fCNN/data/labels/ALL-TAVOR_name-file.txt")
    parser.add_argument('--structures', nargs=2, default=["CortexLeft", "CortexRight"])
    return vars(parser.parse_args())


def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def get_metric_data_for_metric_names(target_image, metric_names, structure_names, subject=None):
    try:
        return get_metric_data([target_image], [metric_names], structure_names, subject)
    except ValueError:
        _metric_names = [metric_name.split(" ")[-1] for metric_name in metric_names]
        return get_metric_data([target_image], [_metric_names], structure_names, subject)


def compute_correlation_row(predicted_fn, target_fns, metric_names, structure_names, verbose=False):
    if verbose:
        print(predicted_fn)
    target_image = nib.load(predicted_fn)
    predicted_data = get_metric_data_for_metric_names(target_image, metric_names, structure_names, None)
    row = list()
    for fn in target_fns:
        row.append(compute_correlation(target_fn=fn, predicted_data=predicted_data, metric_names=metric_names,
                                       structure_names=structure_names))
    return row


def compute_correlation(target_fn, predicted_data, metric_names, structure_names):
    target_image = nib.load(target_fn)
    target_data = get_metric_data_for_metric_names(target_image, metric_names, structure_names, None)
    return pearsonr(predicted_data.flatten(), target_data.flatten())


def main():
    args = parse_args()
    subjects = load_json(args["subjects_filename"])[args["group"]]
    test_template = os.path.join(args["test_dir"], "{subject}", args["template"])
    retest_template = os.path.join(args["retest_dir"], "{subject}", args["template"])
    test_filenames = list()
    retest_filenames = list()
    for subject in subjects:
        test_filename = test_template.format(subject=subject)
        retest_filename = retest_template.format(subject=subject)
        if os.path.exists(test_filename) and os.path.exists(retest_filename):
            test_filenames.append(test_filename)
            retest_filenames.append(retest_filename)

    metric_names = read_namefile(args["metric_names"])

    if args["nthreads"] > 1:
        func = partial(compute_correlation_row, target_fns=retest_filenames, metric_names=metric_names,
                       structure_names=args["structures"], verbose=True)
        pool = Pool(args["nthreads"])
        correlations = pool.map(func, test_filenames)
    else:
        correlations = list()
        for i, test_filename in enumerate(test_filenames):
            update_progress(i/len(retest_filenames), message=os.path.basename(test_filename).split("_")[0])
            correlations.append(compute_correlation_row(test_filename, retest_filenames, metric_names,
                                                        args["structures"]))
        update_progress(1)
    np.save(args["output_filename"], correlations)
    np.save(args["output_filename"].replace(".npy", "_subjects.npy"), subjects)


if __name__ == "__main__":
    main()
