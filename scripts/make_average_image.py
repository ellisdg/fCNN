import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="training")
    parser.add_argument("--config_filename",
                        default="/home/aizenberg/dgellis/fCNN/data/subjects_v4-retest.json")
    return parser.parse_args()


def main():
    import nibabel as nib
    from fcnn.utils.utils import load_json
    from nilearn.image import new_img_like

    namespace = parse_args()

    subjects = load_json(namespace.config_filename)[namespace.group]
    data = None
    filenames = list()
    for subject in sorted(subjects):
        fn = "/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/{0}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.nii.gz".format(subject)
        if os.path.exists(fn):
            filenames.append(fn)
    n = len(filenames)
    print(n)

    for fn in filenames:
        print(fn)
        image = nib.load(fn)
        _data = image.get_fdata()/n
        if data is None:
            data = _data
        else:
            data = data + _data

    new_image = new_img_like(data=data, ref_niimg=image)
    new_image.to_filename("/work/aizenberg/dgellis/fCNN/{}_{}_average_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.nii.gz".format(namespace.group,
                                                                                                                           os.path.basename(namespace.config_filename).split(".")[0]))


if __name__ == "__main__":
    main()
