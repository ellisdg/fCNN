import subprocess
import json
import sys
import os


def load_json(filename):
    with open(filename) as opened_file:
        return json.load(opened_file)


def main(args):
    config = load_json(args[1])
    directory = os.path.abspath(args[2])
    output_directory = os.path.abspath(args[3])
    subset = str(args[4])
    try:
        output_basename = str(args[5])
    except IndexError:
        output_basename = ""
    make_average_cifti_volume(config=config, directory=directory, subset=subset, output_directory=output_directory,
                              output_basename=output_basename)


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


def compute_error_metrics(average_filename, image_filenames, overwrite=False, axis=(0, 1, 2)):
    import nibabel as nib
    import numpy as np
    average_image = nib.load(average_filename)
    for fn in image_filenames:
        error_filename = fn.replace(".nii", "_error.nii")
        if overwrite or not os.path.exists(error_filename):
            image = nib.load(fn)
            error_data = image.get_fdata() - average_image.get_fdata()
            error_image = image.__class__(data=error_data, affine=image.affine)
            error_image.to_filename(error_filename)
        else:
            error_image = nib.load(error_filename)
            error_data = error_image.get_fdata()
        mae = np.mean(np.abs(error_data), axis=axis)
        mse = np.mean(np.square(error_data), axis=axis)



def make_average_cifti_volume(config, directory, output_directory, subset, output_basename, reference_volume=None,
                              overwrite=False):
    if reference_volume is None:
        reference_volume = "/work/aizenberg/dgellis/tools/HCPpipelines/global/templates/MNI152_T1_2mm.nii.gz"
    if subset not in config and "subjects_filename" in config:
        if os.path.exists(config["subjects_filename"]):
            subjects_config_filename = config["subjects_filename"]
        else:
            subjects_config_filename = os.path.join(os.path.dirname(__file__), "..", config["subjects_filename"])
        subjects_config = load_json(subjects_config_filename)
        subject_ids = subjects_config[subset]
    else:
        subject_ids = config[subset]

    target_basenames = config["target_basenames"]
    if type(target_basenames) != list:
        target_basenames = [target_basenames]
    for target_basename in target_basenames:
        make_average_cifti_volume_for_target(target_basename, output_directory, output_basename, subset, subject_ids,
                                             directory, reference_volume, overwrite=overwrite)


def make_average_cifti_volume_for_target(target_basename, output_directory, output_basename, subset, subject_ids,
                                         directory, reference_volume, overwrite=False):
    output_filename = os.path.join(output_directory,
                                   output_basename + os.path.basename(target_basename.format(subset)))
    if overwrite or not os.path.exists(output_filename):
        image_filenames = list()
        for subject_id in subject_ids:
            subject_id = str(subject_id)

            input_volume = os.path.join(directory, subject_id, target_basename.format(subject_id))
            if os.path.exists(input_volume):
                nifti_volume = input_volume.replace(".volume.dscalar", "")
                convert_cmd = ["wb_command", "-cifti-separate", input_volume, "COLUMN", "-volume-all", nifti_volume]
                if overwrite or not os.path.exists(nifti_volume):
                    run_command(convert_cmd)

                warpfield = os.path.join(directory, subject_id, "MNINonLinear", "xfms", "acpc_dc2standard.nii.gz")
                output_volume = nifti_volume.replace("T1w", "MNINonLinear").replace(".nii", "_resampled.nii")
                resample_cmd = ["wb_command", "-volume-warpfield-resample", nifti_volume, warpfield, reference_volume,
                                "TRILINEAR", output_volume, "-fnirt", reference_volume]
                if overwrite or not os.path.exists(output_volume):
                    run_command(resample_cmd)

                image_filenames.append(output_volume)
        compute_average_image(image_filenames, output_filename)
    return output_filename


if __name__ == "__main__":
    main(sys.argv)
