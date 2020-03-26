from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import nibabel as nib
import numpy as np
import sys
from fcnn.utils.utils import load_json, update_progress


def compute_regression_weights(x, y, normalize_=False):
    model = LinearRegression(normalize=normalize_)
    model.fit(x, y)
    return np.concatenate([model.coef_, model.intercept_[..., None]], axis=1)


def fetch_subject_data(subject, feature_template, target_template):
    return fetch_data(feature_template.format(subject=subject).split(",")), \
           fetch_data(target_template.format(subject=subject))


def fetch_data(filenames):
    data = list()
    for filename in filenames:
        print(filename)
        data.append(np.asarray(nib.load(filename).dataobj))
    for a in data:
        print(a.shape)
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
            subject_weights.append(compute_regression_weights(normalize(x), y, normalize_=False))
        except FileNotFoundError:
            pass
    update_progress(1)
    np.save(weights_filename, subject_weights)
    np.save(weights_filename_average, np.mean(subject_weights, axis=0))


if __name__ == "__main__":
    main()
