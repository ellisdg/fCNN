import seaborn
import matplotlib.pyplot as plt
import glob
import os
from fcnn.utils.utils import load_json, update_progress
from fcnn.utils.hcp import get_metric_data, extract_gifti_surface_vertices
from scipy.spatial.distance import cdist
import nibabel as nib
import numpy as np
import sys


def average_around_index(data, surface_vertices, ind, radius=10, distance_metric="euclidean"):
    peak_point = surface_vertices[ind]
    distances = np.squeeze(cdist(surface_vertices, np.asarray([peak_point]), distance_metric))
    mask = distances <= radius
    return np.mean(data[mask])


def compute_lateral_peak(data, surface_vertices, ind=None, radius=10, distance_metric="euclidean",
                         return_index=False):
    assert len(data) == len(surface_vertices)
    if ind is None:
        ind = np.argmax(data)
    avg = average_around_index(data, surface_vertices, ind=ind, radius=radius, distance_metric=distance_metric)
    if return_index:
        return avg, ind
    else:
        return avg


def compute_lateralization_index_from_loaded_data(lh_data, lh_vertices, rh_data, rh_vertices, radius=10,
                                                  distance_metric="euclidean", lh_ind=None, rh_ind=None,
                                                  return_indices=False):
    lh_peak, lh_ind = compute_lateral_peak(lh_data, lh_vertices, radius=radius, distance_metric=distance_metric,
                                           ind=lh_ind, return_index=True)
    rh_peak, rh_ind = compute_lateral_peak(rh_data, rh_vertices, radius=radius, distance_metric=distance_metric,
                                           ind=rh_ind, return_index=True)
    if return_indices:
        return lh_peak - rh_peak, (lh_ind, rh_ind)
    else:
        return lh_peak - rh_peak


def compute_lateralization_index(dscalar, lh_surface, rh_surface, metric_name, subject_id=None, radius=10,
                                 distance_metric="euclidean", lh_ind=None, rh_ind=None, return_indices=False):
    try:
        lh_data = np.squeeze(get_metric_data([dscalar], [[metric_name]], surface_names=["CortexLeft"],
                                             subject_id=subject_id))
    except ValueError:
        metric_name = metric_name.split(" ")[-1]
        lh_data = np.squeeze(get_metric_data([dscalar], [[metric_name]], surface_names=["CortexLeft"],
                                             subject_id=subject_id))

    lh_vertices = extract_gifti_surface_vertices(lh_surface, primary_anatomical_structure="CortexLeft")
    rh_data = np.squeeze(get_metric_data([dscalar], [[metric_name]], surface_names=["CortexRight"],
                                         subject_id=subject_id))
    rh_vertices = extract_gifti_surface_vertices(rh_surface, primary_anatomical_structure="CortexRight")
    return compute_lateralization_index_from_loaded_data(lh_data, lh_vertices, rh_data, rh_vertices, radius=radius,
                                                         distance_metric=distance_metric, lh_ind=lh_ind, rh_ind=rh_ind,
                                                         return_indices=return_indices)


def compute_lateralization_index_at_common_max(dscalar1, dscalar2, lh_surface, rh_surface, metric_name, subject_id=None,
                                               radius=10, distance_metric="euclidean"):
    lateralization1, (lh_ind, rh_ind) = compute_lateralization_index(dscalar=dscalar1, lh_surface=lh_surface,
                                                                     rh_surface=rh_surface, metric_name=metric_name,
                                                                     subject_id=subject_id, radius=radius,
                                                                     distance_metric=distance_metric,
                                                                     return_indices=True)
    lateralization2 = compute_lateralization_index(dscalar=dscalar2, lh_surface=lh_surface, rh_surface=rh_surface,
                                                   metric_name=metric_name, subject_id=subject_id, radius=radius,
                                                   distance_metric=distance_metric, return_indices=False, lh_ind=lh_ind,
                                                   rh_ind=rh_ind)
    return lateralization1, lateralization2


def compute_lateralization_index_from_filenames(dscalar_filename, lh_surface_filename, rh_surface_filename, metric_name,
                                                subject_id=None, radius=10, distance_metric="euclidean"):
    return compute_lateralization_index(nib.load(dscalar_filename), nib.load(lh_surface_filename),
                                        nib.load(rh_surface_filename), metric_name=metric_name, subject_id=subject_id,
                                        radius=radius, distance_metric=distance_metric)


def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def main():
    seaborn.set_palette('muted')
    seaborn.set_style('whitegrid')
    task = sys.argv[1]
    fcnn_dir = "/home/aizenberg/dgellis/fCNN"
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    prediction_dir = sys.argv[2]
    output_dir = sys.argv[3]
    try:
        config_filename = sys.argv[4]
    except IndexError:
        config_filename = fcnn_dir + "/data/v4_struct6_unet_{task}-TAVOR_2mm_v1_pt_config.json".format(task=task)

    config = load_json(config_filename)
    target_basename = config["target_basenames"]
    all_prediction_images = glob.glob(os.path.join(prediction_dir, "*.dscalar.nii"))
    target_images = list()
    hemispheres = ["L", "R"]
    surface_template = "T1w/fsaverage_LR32k/{subject}.{hemi}.{surf}.32k_fs_LR.surf.gii"
    all_surfaces = list()
    prediction_images = list()
    surf_name = "midthickness"
    metric_filename = fcnn_dir + "/data/labels/{task}-TAVOR_name-file.txt".format(task=task)
    metric_names = read_namefile(metric_filename)
    subjects = list()

    for p_image_fn in all_prediction_images:
        if "target" not in p_image_fn:
            sid = os.path.basename(p_image_fn).split("_")[0]
            target_fn = os.path.join(hcp_dir, sid, target_basename.format(sid)).replace(".nii.gz",
                                                                                        ".{}.dscalar.nii").format(
                surf_name)
            target_images.append(target_fn)
            prediction_images.append(p_image_fn)
            all_surfaces.append([os.path.join(hcp_dir, sid, surface_template.format(subject=sid, hemi=hemi,
                                                                                    surf=surf_name))
                                 for hemi in hemispheres])
            subjects.append(sid)

    seaborn.set_style('white')

    lateralization_file = os.path.join(prediction_dir, "lateralization.npy")
    if not os.path.exists(lateralization_file):
        lateralization = list()
        for i, pred_filename, target_filename, surfs, sid in zip(np.arange(len(subjects)), prediction_images,
                                                                 target_images, all_surfaces, subjects):
            row = list()
            pred_dscalar = nib.load(pred_filename)
            target_dscalar = nib.load(target_filename)
            lh_surface = nib.load(surfs[0])
            rh_surface = nib.load(surfs[1])
            for metric_name in metric_names:
                row.append(compute_lateralization_index_at_common_max(pred_dscalar, target_dscalar, lh_surface,
                                                                      rh_surface, metric_name))
            lateralization.append(row)
            update_progress((i + 1)/len(subjects))
        lateralization = np.asarray(lateralization)
        np.save(lateralization_file, lateralization)
        np.save(lateralization_file.replace(".npy", "_subjects.npy"), subjects)


if __name__ == "__main__":
    main()
