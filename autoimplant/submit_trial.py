from slurm.submit_trial import divide_into_folds, dump_json, load_json, submit_slurm_trial, parse_args, os


def submit_cross_validation_trials(config_filename, n_folds, group="training", **slurm_kwargs):
    config = load_json(config_filename)
    subject_ids = ["{:03d}".format(i) for i in range(100)]
    folds = divide_into_folds(subject_ids, n_folds, shuffle=True)
    for i, (train, validation) in enumerate(folds):
        real_train = list()
        real_validation = list()
        for s1 in train:
            for s2 in train:
                real_train.append("sub-{:03d}_space-{:03d}".format(s1, s2))
        for s1 in validation:
            real_validation.append("sub-{0:03d}_space-{0:03d}".format(s1))
        config["training"] = real_train
        config["validation"] = real_validation
        fold_config_filename = config_filename.replace(".json", "_fold{}.json".format(i))
        dump_json(config, fold_config_filename)
        submit_slurm_trial(fold_config_filename, **slurm_kwargs)


def main():
    namespace = parse_args()
    assert os.path.exists(namespace.trial_config)
    slurm_config = load_json(namespace.slurm_config)
    if namespace.n_folds > 1:
        submit_cross_validation_trials(namespace.trial_config, namespace.n_folds, output_dir=namespace.output_dir,
                                       **slurm_config)
    else:
        submit_slurm_trial(namespace.trial_config, output_dir=namespace.output_dir, **slurm_config)


if __name__ == "__main__":
    main()
