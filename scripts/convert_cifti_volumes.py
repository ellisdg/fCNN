import subprocess
import json
import sys
import os
import nibabel as nib
import numpy as np
from fcnn.utils.hcp import extract_cifti_scalar_map_names


def load_json(filename):
    with open(filename) as opened_file:
        return json.load(opened_file)


def main(args):
    config = load_json(args[1])
    directory = os.path.abspath(args[2])
    basename = str(args[3])
    crop = True
    overwrite = False
    subjects_config_filename = os.path.join(os.path.dirname(__file__), "..", config["subjects_filename"])
    subjects_config = load_json(subjects_config_filename)
    target_basenames = config["target_basenames"]
    all_metric_names = config["metric_names"]
    if type(target_basenames) != list:
        target_basenames = [target_basenames]
        all_metric_names = [all_metric_names]
    for subject_ids in subjects_config.values():
        for target_basename, metric_names in zip(target_basenames, all_metric_names):
            prep_cifti_volumes(subjects=subject_ids, directory=directory, target_basename=target_basename,
                               metric_names=metric_names, basename=basename, overwrite=overwrite, crop=crop)


def run_command(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)


def compute_average_image(image_filenames, output_filename):
    import nibabel as nib
    data = None
    ref_image = None
    for fn in image_filenames:
        image = nib.load(fn)
        if data is None:
            data = image.get_fdata()/len(image_filenames)
            ref_image = image
        else:
            data[:] += image.get_fdata()/len(image_filenames)
    average = ref_image.__class__(data, ref_image.affine)
    average.to_filename(output_filename)
    return output_filename


def prep_cifti_volumes(subjects, directory, target_basename, basename, metric_names, overwrite=False, crop=True):
    for subject_id in subjects:
        cifti_volume = os.path.join(directory, subject_id, target_basename.format(subject_id))
        nifti_volume = cifti_volume.replace(".volume.dscalar.nii", ".nii.gz")
        if overwrite or not os.path.exists(nifti_volume):
            convert_cifti_volume(cifti_volume, nifti_volume, crop=crop)
        output_volume = nifti_volume.replace(".nii", "_{}.nii".format(basename))
        if overwrite or not os.path.exists(output_volume):
            nifti_labels = read_labels(cifti_volume)
            target_labels = [name.format(subject_id) for name in metric_names]
            slice_nifti_volume(nifti_volume, nifti_labels, target_labels, output_volume)


def slice_nifti_volume(input_volume, input_labels, target_labels, output_volume):
    volume = nib.load(input_volume)
    sliced_volume = slice_volume(volume, input_labels, target_labels)
    sliced_volume.to_filename(output_volume)


def slice_volume(volume, labels, target_labels):
    data = np.asarray(volume.dataobj)
    mask = np.in1d(labels, target_labels)
    return volume.__class__(dataobj=data[..., mask], affine=volume.affine)


def read_labels(cifti_volume):
    return extract_cifti_scalar_map_names(nib.load(cifti_volume))


def convert_cifti_volume(cifti_volume, output_volume, direction="COLUMN", crop=True):
    cmd = ["wb_command", "-cifti-separate", cifti_volume, direction, "-volume-all", output_volume]
    if crop:
        cmd.append("-crop")
    run_command(cmd)
    return output_volume


if __name__ == "__main__":
    main(sys.argv)
