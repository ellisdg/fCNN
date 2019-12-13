import os
import sys
import tables
import nibabel as nib
import numpy as np
from fcnn.utils.hcp import get_metric_data


def cifti2h5(input_filenames, output_filename, metric_names, surface_names):
    table = tables.open_file(output_filename, mode="a")
    subject_ids = list()
    data = list()
    for fn in input_filenames:
        sid, group_name, _ = os.path.basename(fn).split("_", 2)
        subject_ids.append(sid)
        cifti_image = nib.load(fn)
        get_metric_data([cifti_image], metric_names=None, surface_names=None, subject_id=sid)
        data.append(None)


def main():
    output_filename = sys.argv[0]
    input_filenames = sys.argv[1:]


if __name__ == "__main__":
    main()
