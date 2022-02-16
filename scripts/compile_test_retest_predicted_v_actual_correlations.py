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
    parser.add_argument('--test_pred_template',
                        default="/work/aizenberg/dgellis/fCNN/predictions/"
                                "v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test-retest_test/"
                                "{subject}_model_v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_struct14_normalized."
                                "midthickness.dscalar.nii")
    parser.add_argument('--retest_pred_template',
                        default="/work/aizenberg/dgellis/fCNN/predictions/"
                                "v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test-retest_retest/"
                                "{subject}_model_v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_struct14_normalized."
                                "midthickness.dscalar.nii")
    parser.add_argument('--subjects_filename', default="/home/aizenberg/dgellis/fCNN/data/subjects_v4-retest.json")
    parser.add_argument('--group', default="retest")
    parser.add_argument('--template',
                        default="T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/"
                                "{subject}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii")
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--metric_names', default="/home/aizenberg/dgellis/fCNN/data/labels/ALL-TAVOR_name-file.txt")
    parser.add_argument('--structures', nargs=2, default=["CortexLeft", "CortexRight"])
    parser.add_argument('--verbose', action='store_true', default=False)
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


def compute_correlation_row(filenames, metric_names, structure_names, verbose=False):
    if verbose:
        print(" ".join(filenames))
    test, retest, p_test, p_retest = [get_metric_data_for_metric_names(nib.load(fn), metric_names, structure_names,
                                                                       None)
                                      for fn in filenames]
    # compute test-retest correlation
    r_test_retest = pearsonr(test.flatten(), retest.flatten())
    # compute predicted test to actual retest correlation
    r_p_test_retest = pearsonr(p_test.flatten(), retest.flatten())
    # compute actual test to predicted retest correlation
    r_test_p_retest = pearsonr(test.flatten(), p_retest.flatten())
    return [r_test_retest, r_p_test_retest, r_test_p_retest]


def main():
    args = parse_args()
    subjects = load_json(args["subjects_filename"])[args["group"]]
    test_template = os.path.join(args["test_dir"], "{subject}", args["template"])
    retest_template = os.path.join(args["retest_dir"], "{subject}", args["template"])
    filenames = list()
    for subject in subjects:
        test_filename = test_template.format(subject=subject)
        retest_filename = retest_template.format(subject=subject)
        pred_test_filename = args["test_pred_template"].format(subject=subject)
        pred_retest_filename = args["retest_pred_template"].format(subject=subject)
        _filenames = (test_filename, retest_filename, pred_test_filename, pred_retest_filename)
        if all([os.path.exists(filename) for filename in _filenames]):
            filenames.append(_filenames)
        else:
            for filename in _filenames:
                if not os.path.exists(filename):
                    print("Does not exist:", filename)

    metric_names = read_namefile(args["metric_names"])

    if args["nthreads"] > 1:
        func = partial(compute_correlation_row, metric_names=metric_names, structure_names=args["structures"],
                       verbose=args["verbose"])
        pool = Pool(args["nthreads"])
        correlations = pool.map(func, filenames)
    else:
        correlations = list()
        for i, _filenames in enumerate(filenames):
            update_progress(i/len(filenames), message=os.path.basename(_filenames[0]).split("_")[0])
            correlations.append(compute_correlation_row(filenames=_filenames, metric_names=metric_names,
                                                        structure_names=args["structures"]))
        update_progress(1)
    np.save(args["output_filename"], correlations)
    np.save(args["output_filename"].replace(".npy", "_subjects.npy"), subjects)


if __name__ == "__main__":
    main()
