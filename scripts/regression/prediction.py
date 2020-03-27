from sklearn.preprocessing import normalize
import nibabel as nib
import numpy as np
import sys
from fcnn.utils.utils import load_json, update_progress
from fcnn.utils.hcp import new_cifti_scalar_exactly_like, get_vertices_from_scalar, get_mask_from_axis, get_axis


def predict(x, weights):
    return np.matmul(weights, np.concatenate([x, np.ones(x.shape[1])[None]], axis=0))


def fetch_subject_data(subject, feature_template, target_template):
    return fetch_data(feature_template.format(subject=subject).split(","),
                      target_template.format(subject=subject).split(","))


def create_predicted_dscalar():
    """
    The predicted dscalar has lower number of points than the target dscalar. This functions takes the predicted data
    and creates a
    :return:
    """


def fetch_data(filenames, target_filenames, structure_names=("CortexLeft", "CortexRight")):
    dscalars = list()
    vertices = dict()
    for filename in filenames + target_filenames:
        dscalar = nib.load(filename)
        dscalars.append(dscalar)
        for structure_name in structure_names:
            dscalar_structure_vertices = get_vertices_from_scalar(dscalar, structure_name)
            if structure_name in vertices:
                vertices[structure_name] = np.intersect1d(dscalar_structure_vertices, vertices[structure_name])
            else:
                vertices[structure_name] = dscalar_structure_vertices
    data = list()
    target_data = list()
    for i, dscalar in enumerate(dscalars):
        dscalar_data = list()
        mask = np.zeros(dscalar.shape[-1], np.bool)
        for structure_name, _vertices in vertices.items():
            brain_model_axis = get_axis(dscalar, axis_index=1)
            structure_mask = get_mask_from_axis(brain_model_axis, structure_name)
            dscalar_structure_vertices = brain_model_axis.vertex[structure_mask]
            hemi_mask = np.isin(dscalar_structure_vertices, _vertices)
            dscalar_data.append(np.asarray(dscalar.dataobj)[..., structure_mask][..., hemi_mask])
            mask[structure_mask] = hemi_mask
        dscalar_data = np.concatenate(dscalar_data, axis=1)
        if i < len(filenames):
            data.append(dscalar_data)
        else:
            target_data.append(dscalar_data)
            target_mask = mask

    return (np.concatenate(data, axis=0), np.swapaxes(np.concatenate(target_data, axis=0), 0, 1), target_mask)


def main():
    key = "validation"
    config_filename = sys.argv[1]
    weights_filename = sys.argv[2]
    feature_template = sys.argv[3]
    target_template = sys.argv[4]
    output_template = sys.argv[5]
    structure_names = ["CortexLeft", "CortexRight"]
    config = load_json(config_filename)
    weights = np.load(weights_filename)
    for i, subject in enumerate(config[key]):
        update_progress(i/len(config[key]), message=str(subject))
        try:
            x, y, mask = fetch_subject_data(subject, feature_template, target_template)
            target_dscalar = nib.load(target_template.format(subject=subject))
        except FileNotFoundError:
            continue
        pred_y = predict(normalize(x), weights)
        pred_full = np.zeros(target_dscalar.shape)
        pred_full[..., mask] = pred_y
        predicted_dscalar = new_cifti_scalar_exactly_like(pred_full, structure_names, target_dscalar)
        output_filename = output_template.format(subject=subject)
        predicted_dscalar.to_filename(output_filename)

    update_progress(1)


if __name__ == "__main__":
    main()
