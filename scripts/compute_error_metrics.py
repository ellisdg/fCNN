import json
import sys
import os


def load_json(filename):
    with open(filename) as opened_file:
        return json.load(opened_file)


def main(args):
    config = load_json(args[1])
    directory = os.path.abspath(args[2])
    average_filename = os.path.abspath(args[3])
    subset = str(args[4])
    basename = str(args[5])
    if subset not in config and "subjects_filename" in config:
        if os.path.exists(config["subjects_filename"]):
            subjects_config_filename = config["subjects_filename"]
        else:
            subjects_config_filename = os.path.join(os.path.dirname(__file__), "..", config["subjects_filename"])
        subjects_config = load_json(subjects_config_filename)
        subject_ids = subjects_config[subset]
    else:
        subject_ids = config[subset]

    output_directory = os.path.dirname(average_filename)
    basename_basename = os.path.basename(basename)
    output_template = os.path.join(output_directory, basename_basename).format(subset)
    compute_error_metrics(average_filename=average_filename, directory=directory, subject_ids=subject_ids,
                          basename=basename,
                          output_mae_filename=output_template.replace(".nii", "_mae.csv"),
                          output_mse_filename=output_template.replace(".nii", "_mse.csv"))


def compute_error_metrics(average_filename, subject_ids, directory, basename, output_mae_filename, output_mse_filename,
                          overwrite=False, axis=(0, 1, 2)):
    import nibabel as nib
    import numpy as np
    import pandas as pd

    average_image = nib.load(average_filename)
    mae_rows = list()
    mse_rows = list()
    index = list()
    for subject_id in subject_ids:
        fn = os.path.join(directory, subject_id, basename.format(subject_id))
        if os.path.exists(fn):
            index.append(subject_id)
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
            mae_rows.append(mae)
            mse = np.mean(np.square(error_data), axis=axis)
            mse_rows.append(mse)
    mae_df = pd.DataFrame(data=mae_rows, index=index)
    mae_df.to_csv(output_mae_filename)
    mse_df = pd.DataFrame(data=mse_rows, index=index)
    mse_df.to_csv(output_mse_filename)
    return mae_df, mse_df


if __name__ == "__main__":
    main(sys.argv)
