import os
import nibabel as nib
from fcnn.utils.hcp import get_metric_data
from fcnn.utils.utils import load_json
import numpy as np
import torch


def read_anat_features(subject):
    row_data = list()
    for feature in ("thickness", "MyelinMap", "sulc", "curvature"):
        fn = "/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/fsaverage_LR32k/{0}.{1}.32k_fs_LR.dscalar.nii".format(
            subject, feature
        )
        print(fn)
        image = nib.load(fn)
        row_data.append(
            get_metric_data(
                [image],
                [[index.map_name for index in image.header.get_index_map(0)]],
                ["CORTEX_LEFT", "CORTEX_RIGHT"],
                None,
            )
        )
    return row_data


def read_subject(subject, data, fmri_data, s_training, anat_data):
    fn = struct14_wc.format(subject)
    fmri_fn = fmri_wc.format(subject)
    if os.path.exists(fn) and os.path.exists(fmri_fn):
        print("loading,", subject)
        s14 = nib.load(fn)
        data.append(
            get_metric_data([s14], [struct14_map_names], ["CORTEX_LEFT", "CORTEX_RIGHT"], None)
        )
        read_fmri(fmri_fn, fmri_data)
        anat_data.append(read_anat_features(subject))
        s_training.append(subject)


def read_fmri(fmri_fn, fmri_data):
    fmri = nib.load(fmri_fn)
    try:
        fmri_data.append(
            get_metric_data(
                [fmri],
                [fmri_map_names_short],
                ["CORTEX_LEFT", "CORTEX_RIGHT"],
                None,
            )
        )
    except ValueError:
        fmri_data.append(
            get_metric_data(
                [fmri], [fmri_map_names_long], ["CORTEX_LEFT", "CORTEX_RIGHT"], None
            )
        )


struct14_wc = "/work/aizenberg/dgellis/HCP/HCP_1200/{}/T1w/struct14_normalized.midthickness.dscalar.nii"
fmri_wc = "/work/aizenberg/dgellis/HCP/HCP_1200/{0}/T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/{0}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii"
struct14_map_names = ['T1w',
             'T2w',
             'MD-1000',
             'FA-R-1000',
             'FA-G-1000',
             'FA-B-1000',
             'MD-21000',
             'FA-R-2000',
             'FA-G-2000',
             'FA-B-2000',
             'MD-3000',
                      'FA-R-3000',
                      'FA-G-3000',
                      'FA-B-3000']

fmri_map_names_short = ['CUE',
                        'LF',
                        'LH',
                        'RF',
                        'RH',
                        'T',
                        'AVG',
                        'CUE-AVG',
                        'LF-AVG',
                        'LH-AVG',
                        'RF-AVG',
                        'RH-AVG',
                        'T-AVG',
                        'STORY',
                        'MATH',
                        'MATH-STORY',
                        '2BK_BODY',
                        '2BK_FACE',
                        '2BK_PLACE',
                        '2BK_TOOL',
                        '0BK_BODY',
                        '0BK_FACE',
                        '0BK_PLACE',
                        '0BK_TOOL',
                        '2BK',
                        '0BK',
                        '2BK-0BK',
                        'BODY',
                        'FACE',
                        'PLACE',
                        'TOOL',
                        'BODY-AVG',
                        'FACE-AVG',
                        'PLACE-AVG',
                        'TOOL-AVG',
                        'REL',
                        'MATCH',
                        'MATCH-REL',
                        'FACES',
                        'SHAPES',
                        'FACES-SHAPES',
                        'TOM',
                        'RANDOM',
                        'RANDOM-TOM',
                        'PUNISH',
                        'REWARD',
                        'PUNISH-REWARD']

fmri_map_names_long = ['MOTOR CUE',
                       'MOTOR LF',
                       'MOTOR LH',
                       'MOTOR RF',
                       'MOTOR RH',
                       'MOTOR T',
                       'MOTOR AVG',
                       'MOTOR CUE-AVG',
                       'MOTOR LF-AVG',
                       'MOTOR LH-AVG',
                       'MOTOR RF-AVG',
                       'MOTOR RH-AVG',
                       'MOTOR T-AVG',
                       'LANGUAGE MATH',
                       'LANGUAGE STORY',
                       'LANGUAGE MATH-STORY',
                       'WM 2BK_BODY',
                       'WM 2BK_FACE',
                       'WM 2BK_PLACE',
                       'WM 2BK_TOOL',
                       'WM 0BK_BODY',
                       'WM 0BK_FACE',
                       'WM 0BK_PLACE',
                       'WM 0BK_TOOL',
                       'WM 2BK',
                       'WM 0BK',
                       'WM 2BK-0BK',
                       'WM BODY',
                       'WM FACE',
                       'WM PLACE',
                       'WM TOOL',
                       'WM BODY-AVG',
                       'WM FACE-AVG',
                       'WM PLACE-AVG',
                       'WM TOOL-AVG',
                       'RELATIONAL REL',
                       'RELATIONAL MATCH',
                       'RELATIONAL MATCH-REL',
                       'EMOTION FACES',
                       'EMOTION SHAPES',
                       'EMOTION FACES-SHAPES',
                       'SOCIAL TOM',
                       'SOCIAL RANDOM',
                       'SOCIAL RANDOM-TOM',
                       'GAMBLING PUNISH',
                       'GAMBLING REWARD',
                       'GAMBLING PUNISH-REWARD']


def load_roi(subject="100307"):
    l_roi = nib.load("/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/fsaverage_LR32k/{0}.L.atlasroi.32k_fs_LR.shape.gii".format(subject))
    r_roi = nib.load("/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/fsaverage_LR32k/{0}.R.atlasroi.32k_fs_LR.shape.gii".format(subject))
    roi = np.concatenate([l_roi.darrays[0].data, r_roi.darrays[0].data])
    return roi


def load_data(subjects, norm_features=True, output_dir="/work/aizenberg/dgellis/fCNN/regression", output_prefix=""):
    feature_fn = os.path.join(output_dir, output_prefix + "features.npy")
    fmri_fn = os.path.join(output_dir, output_prefix + "targets.npy")
    subjects_fn = os.path.join(output_dir, output_prefix + "subjects.npy")
    if not any([os.path.exists(fn) for fn in (feature_fn, fmri_fn, subjects_fn)]):
        struct14_data = list()
        fmri_data = list()
        select_subjects = list()
        anat_data = list()
        for subject in subjects:
            read_subject(subject, struct14_data, fmri_data, select_subjects, anat_data)
        # now that we have collected the data, a few things need to be cleaned up
        # the struct14, sulc, and fmri data have vertices that aren't in the roi
        # so we will take them out now
        roi = load_roi()
        roi_mask = roi == 1
        struct14_data = np.asarray(struct14_data)[:, roi_mask]
        anat_data_final = list()
        for row in anat_data:
            anat_data_final.append([row[0], row[1], row[2][roi_mask], row[3]])
        fmri_data = np.asarray(fmri_data)[:, roi_mask]
        # now we have all our data
        # let's combine the anat and struct14 data
        features = np.concatenate([struct14_data] + [np.asarray([row[i] for row in anat_data_final]) for i in range(4)],
                                  axis=-1)
        np.save(feature_fn, features)
        np.save(fmri_fn, fmri_data)
        np.save(subjects_fn, select_subjects)
    else:
        features = np.load(feature_fn)
        fmri_data = np.load(fmri_fn)
        select_subjects = np.load(subjects_fn)
    if norm_features:
        # normalize the features by the values for the whole brain
        features = (features - features.mean(axis=1)[:, None]) / features.std(axis=1)[:, None]
    return features, fmri_data, np.asarray(select_subjects)


def to_torch_features(training_features):
    A_np = np.concatenate([training_features,
                           np.ones(training_features.shape[:2])[..., None]], axis=-1)
    A = torch.Tensor(A_np)
    return A


def main(cuda=0):
    subjects_config = load_json("/home/aizenberg/dgellis/fCNN/data/subjects_v4.json")
    X = fit_model(initial_training_subjects=subjects_config["training"]).cuda(cuda)
    test_features, _, test_subjects = load_data(subjects_config["test"], output_prefix="test_")
    A = to_torch_features(test_features).cuda(cuda)
    B = (A[..., None] * X[None]).sum(dim=-2)
    torch.save(B, "/work/aizenberg/dgellis/fCNN/regression/B_pointwise_60k_test_prediction.pt")


def fit_model(initial_training_subjects,
              X_filename="/work/aizenberg/dgellis/fCNN/regression/X_pointwise_60k_solution.pt",
              overwrite=False):
    if not os.path.exists(X_filename) or overwrite:
        training_features, training_target, training_subjects = load_data(initial_training_subjects)

        # we are going to add 1 to n_features to fit the intercept
        # we are also going to swap the axis so that the model fits every individual point
        A = to_torch_features(training_features).cuda()
        B_np = training_target.swapaxes(0, 1)
        B = torch.Tensor(B_np).cuda()
        X = torch.linalg.lstsq(A, B).solution
        torch.save(X, X_filename)
    else:
        X = torch.load(X_filename).cuda()
    return X


if __name__ == "__main__":
    main()
