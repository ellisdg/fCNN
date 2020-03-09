import sys
import os
import nibabel as nib
import numpy as np
from fcnn.utils.hcp import get_metric_data
from fcnn.utils.utils import load_json


def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def main():
    dscalar_template = sys.argv[1]
    name_file = sys.argv[2]
    config = load_json(sys.argv[3])
    output_file = sys.argv[4]
    key = sys.argv[5]
    directory = sys.argv[6]
    data = list()
    metric_names = read_namefile(name_file)
    structure_names = ["CortexLeft", "CortexRight"]
    subjects = list()
    for subject_id in sorted(config[key]):
        dscalar_file = dscalar_template.format(directory=directory, subject=subject_id)
        if os.path.exists(dscalar_file):
            data.append(get_metric_data([nib.load(dscalar_file)], [metric_names], surface_names=structure_names,
                                        subject_id=subject_id))
            subjects.append(subject_id)
    data = np.asarray(data)
    np.save(output_file, data)
    np.save(output_file.replace(".", "_subjects."), subjects)


def compile_dscalar_data(dscalar_files, metric_names, structure_names, subject_ids=None):
    data = list()
    for index, dscalar_file in enumerate(dscalar_files):
        if subject_ids is not None:
            subject_id = subject_ids[index]
        else:
            subject_id = None
        data.append(get_metric_data([nib.load(dscalar_file)], [metric_names], surface_names=structure_names,
                                    subject_id=subject_id))
    return np.asarray(data)


if __name__ == "__main__":
    main()
