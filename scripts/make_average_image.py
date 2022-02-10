def main():
    import os
    import nibabel as nib
    from fcnn.utils.utils import load_json
    from nilearn.image import new_img_like

    subjects = load_json("/home/aizenberg/dgellis/fCNN/data/subjects_v4-retest.json")["training"]
    data = None
    filenames = list()
    for subject in sorted(subjects):
        fn1 = "/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/Results/tfMRI_ALL/{0}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.nii.gz".format(subject)
        fn2 = "/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/{0}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.nii.gz".format(subject)
        if os.path.exists(fn1):
            fn = fn1
        elif os.path.exists(fn2):
            fn = fn2
        else:
            continue
        filenames.append(fn)
    n = len(filenames)

    for fn in filenames:
        print(fn)
        image = nib.load(fn)
        _data = image.get_fdata()/n
        if data is None:
            data = _data
        else:
            data = data + _data

    new_image = new_img_like(data=data, ref_niimg=image)
    new_image.to_filename("/work/aizenberg/dgellis/fCNN/v4_training_average_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.nii.gz")


if __name__ == "__main__":
    main()
