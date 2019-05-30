import os
from fcnn.utils.utils import load_json
from fcnn.utils.sequences import load_image


def main():
    config = load_json(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "data",
                                    "t1t2dti4_wb_18_LS_config.json"))
    system_config = load_json(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           "data",
                                           "hcc_p100_config.json"))
    for subject_id in config['validation'] + config["training"]:
        subject_dir = os.path.join(system_config['directory'], subject_id)
        output_filename = os.path.join(subject_dir, "T1w", "struct6.nii.gz")
        if not os.path.exists(output_filename):
            feature_filenames = [os.path.join(subject_dir, fbn) for fbn in config["feature_basenames"]]
            image = load_image(feature_filenames)
            image.to_filename(output_filename)


if __name__ == "__main__":
    main()
