import seaborn
import matplotlib.pyplot as plt
import glob
import os
from fcnn.utils.utils import load_json
from fcnn.utils.hcp import get_metric_data, extract_gifti_surface_vertices
from scipy.spatial.distance import cdist
import nibabel as nib
import numpy as np
import sys


def compute_lateral_peak(data, surface_vertices, radius=10, distance_metric="euclidean"):
    assert len(data) == len(surface_vertices)
    ind = np.argmax(data)
    peak_point = surface_vertices[ind]
    distances = np.squeeze(cdist(surface_vertices, np.asarray([peak_point]), distance_metric))
    mask = distances <= radius
    return np.mean(data[mask])


def compute_lateralization_index_from_loaded_data(lh_data, lh_vertices, rh_data, rh_vertices, radius=10,
                                                  distance_metric="euclidean"):
    lh_peak = compute_lateral_peak(lh_data, lh_vertices, radius=radius, distance_metric=distance_metric)
    rh_peak = compute_lateral_peak(rh_data, rh_vertices, radius=radius, distance_metric=distance_metric)
    return lh_peak - rh_peak


def compute_lateralization_index(dscalar, lh_surface, rh_surface, metric_name, subject_id=None, radius=10,
                                 distance_metric="euclidean"):
    lh_data = np.squeeze(get_metric_data([dscalar], [[metric_name]], surface_names=["CortexLeft"],
                                         subject_id=subject_id))
    lh_vertices = extract_gifti_surface_vertices(lh_surface, primary_anatomical_structure="CortexLeft")
    rh_data = np.squeeze(get_metric_data([dscalar], [[metric_name]], surface_names=["CortexRight"],
                                         subject_id=subject_id))
    rh_vertices = extract_gifti_surface_vertices(rh_surface, primary_anatomical_structure="CortexRight")
    return compute_lateralization_index_from_loaded_data(lh_data, lh_vertices, rh_data, rh_vertices, radius=radius,
                                                         distance_metric=distance_metric)


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
    config_filename = fcnn_dir + "/data/v4_struct6_unet_{task}-TAVOR_2mm_v1_pt_config.json".format(task=task)
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    config = load_json(config_filename)
    target_basename = config["target_basenames"]
    prediction_dir = sys.argv[2]
    output_dir = sys.argv[3]
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
        for pred_filename, target_filename, surfs, sid in zip(prediction_images, target_images, all_surfaces, subjects):
            row = list()
            pred_dscalar = nib.load(pred_filename)
            target_dscalar = nib.load(target_filename)
            lh_surface = nib.load(surfs[0])
            rh_surface = nib.load(surfs[1])
            for metric_name in metric_names:
                row.append([compute_lateralization_index(pred_dscalar, lh_surface, rh_surface, metric_name),
                            compute_lateralization_index(target_dscalar, lh_surface, rh_surface, metric_name)])
            lateralization.append(row)

        lateralization = np.asarray(lateralization)
        np.save(lateralization_file, lateralization)
        np.save(lateralization_file.replace(".npy", "_subjects.npy"), subjects)
    else:
        lateralization = np.load(lateralization_file)

    for task_ind in range(lateralization.shape[-1]):
        fig, ax = plt.subplots(figsize=(18, 6))
        x = np.arange(lateralization.shape[0])
        ind = np.argsort(lateralization[..., 1, task_ind], axis=0)
        ax.bar(x=x, height=lateralization[..., 1, task_ind][ind], width=0.5, label="actual")
        ax.bar(x=x + 0.5, height=lateralization[..., 0, task_ind][ind], width=0.5, label="predicted")
        ax.legend()
        ax.set_title(metric_names[task_ind])
        fig.savefig(output_dir + '/lateralization_{task}_{name}.png'.format(task=task, name=metric_names[task_ind]))


if __name__ == "__main__":
    main()
