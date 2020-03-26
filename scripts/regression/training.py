from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import nibabel as nib
import numpy as np
import sys
from functools import reduce
from fcnn.utils.utils import load_json, update_progress
from fcnn.utils.hcp import get_vertices_from_scalar, get_mask_from_axis, get_axis


def compute_regression_weights(x, y, normalize_=False):
    model = LinearRegression(normalize=normalize_)
    model.fit(x, y)
    return np.concatenate([model.coef_, model.intercept_[..., None]], axis=1)


def fetch_subject_data(subject, feature_template, target_template):
    return fetch_data(feature_template.format(subject=subject).split(",")), \
           fetch_data(target_template.format(subject=subject).split(","))


def fetch_data(filenames, structure_names=("CortexLeft", "CortexRight")):
    dscalars = list()
    vertices = dict()
    for filename in filenames:
        dscalar = nib.load(filename)
        dscalars.append(dscalar)
        for structure_name in structure_names:
            dscalar_structure_vertices = get_vertices_from_scalar(dscalar, structure_name)
            if structure_name in vertices:
                vertices[structure_name] = np.intersect1d(dscalar_structure_vertices, vertices[structure_name])
            else:
                vertices[structure_name] = dscalar_structure_vertices
    data = list()
    for dscalar in dscalars:
        dscalar_data = list()
        for structure_name, _vertices in vertices.items():
            brain_model_axis = get_axis(dscalar, axis_index=1)
            structure_mask = get_mask_from_axis(brain_model_axis, structure_name)
            dscalar_structure_vertices = brain_model_axis.vertex[structure_mask]
            mask = np.isin(dscalar_structure_vertices, _vertices)
            print(dscalar.dataobj.shape, structure_mask.shape, mask.shape)
            dscalar_data.append(np.asarray(dscalar.dataobj)[..., structure_mask][..., mask])
        data.append(np.concatenate(dscalar_data, axis=1))
    return np.swapaxes(np.concatenate(data, axis=0), 0, 1)


def main():
    key = "training"
    config_filename = sys.argv[1]
    weights_filename = sys.argv[2]
    feature_template = sys.argv[3]
    target_template = sys.argv[4]
    weights_filename_average = weights_filename.replace(".npy", "_average.npy")

    config = load_json(config_filename)
    subject_weights = list()
    for i, subject in enumerate(config[key]):
        update_progress(i/len(config[key]), message=str(subject))
        try:
            x, y = fetch_subject_data(subject, feature_template, target_template)
            print(x.shape, y.shape)
            subject_weights.append(compute_regression_weights(normalize(x), y, normalize_=False))
        except FileNotFoundError:
            pass
    update_progress(1)
    np.save(weights_filename, subject_weights)
    np.save(weights_filename_average, np.mean(subject_weights, axis=0))


if __name__ == "__main__":
    main()
