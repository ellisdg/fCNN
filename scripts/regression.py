from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import nibabel as nib
import numpy as np
import sys
from fcnn.utils.utils import load_json, update_progress
from fcnn.scripts.run_trial import load_subject_ids


def compute_regression_weights(x, y, normalize_=False):
    model = LinearRegression(normalize=normalize_)
    model.fit(x, y)
    return np.concatenate(model.coef_, model.intercept_, axis=0)


def fetch_subject_data(subject, feature_template, target_template):
    return fetch_data(feature_template.format(subject=subject)), fetch_data(target_template.format(subject=subject))


def fetch_data(filename):
    return np.asarray(nib.load(filename).dataobj)


def main():
    key = "training"
    config_filename = sys.argv[0]
    weights_filename = sys.argv[1]
    feature_template = sys.argv[2]
    target_template = sys.argv[3]
    weights_filename_average = weights_filename.replace(".npy", "_average.npy")
    config = load_json(config_filename)
    if key not in config:
        load_subject_ids(config)
    subject_weights = list()
    for i, subject in enumerate(config[key]):
        update_progress(i/len(config[key]), message=str(subject))
        x, y = fetch_subject_data(subject, feature_template, target_template)
        subject_weights.append(compute_regression_weights(normalize(x), y, normalize_=False))
    update_progress(1)
    np.save(weights_filename, subject_weights)
    np.save(np.mean(weights_filename_average, subject_weights, axis=0))


if __name__ == "__main__":
    main()
