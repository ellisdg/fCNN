from sklearn.preprocessing import normalize
import nibabel as nib
import numpy as np
import sys
from fcnn.utils.utils import load_json, update_progress
from fcnn.utils.hcp import new_cifti_scalar_exactly_like


def predict(x, weights):
    return np.matmul(weights, np.concatenate([x, np.ones(x.shape[1])[None]], axis=0))


def fetch_data(filename):
    return np.swapaxes(np.asarray(nib.load(filename).dataobj), 0, 1)


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
            dscalar = nib.load(feature_template.format(subject=subject))
            target_dscalar = nib.load(target_template.format(subject=subject))
        except FileNotFoundError:
            continue
        x = np.asarray(dscalar.dataobj)
        y = predict(normalize(x), weights)
        predicted_dscalar = new_cifti_scalar_exactly_like(y, structure_names, target_dscalar)
        output_filename = output_template.format(subject=subject)
        predicted_dscalar.to_filename(output_filename)

    update_progress(1)


if __name__ == "__main__":
    main()
