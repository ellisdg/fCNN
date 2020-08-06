import argparse
import json
import os
import subprocess
from random import shuffle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_config")
    parser.add_argument("--n_folds", default=1, type=int)
    parser.add_argument("--slurm_config",
                        default="/home/aizenberg/dgellis/fCNN/data/hcc_v100_2gpu_32gb_slurm_config.json")
    parser.add_argument("--output_dir", default="/work/aizenberg/dgellis/fCNN/predictions")
    return parser.parse_args()


def load_json(filename):
    with open(filename) as opened_file:
        return json.load(opened_file)


def dump_json(obj, filename):
    with open(filename) as opened_file:
        return json.dump(obj, opened_file)


def load_subject_ids(config):
    fcnn_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if "subjects_filename" in config:
        subjects = load_json(os.path.join(fcnn_path, config["subjects_filename"]))
        for key, value in subjects.items():
            config[key] = value


def submit_slurm_trial(config_filename, job_name=None, partition="gpu", n_gpus=2, constraint="gpu_v100", days=7,
                       n_tasks=40, mem_per_cpu=4000, error_log=None, output_log=None, anaconda_env="fcnn-1.12",
                       fcnn_dir="/home/aizenberg/dgellis/fCNN",
                       python="/home/aizenberg/dgellis/.conda/envs/fcnn-1.12/bin/python",
                       model_filename=None, training_log_filename=None, output_dir=None,
                       machine_config_filename="/home/aizenberg/dgellis/fCNN/data/hcc_v100_2gpu_32gb_config.json"):
    config_basename = os.path.basename(config_filename).split(".")[0].replace("_config", "")
    if job_name is None:
        job_name = config_basename

    if output_dir is None:
        output_dir = os.path.abspath(".")
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_filename is None:
        model_filename = os.path.join(output_dir, "model_" + config_basename + ".h5")

    if training_log_filename is None:
        training_log_filename = os.path.join(output_dir, "log_" + config_basename + ".csv")

    if output_log is None:
        output_log = os.path.join(output_dir, "job." + job_name + "_%J.out".format(job_name))

    if error_log is None:
        error_log = os.path.join(output_dir, "job." + job_name + "_%J.err".format(job_name))

    slurm_script = """#!/bin/sh
#SBATCH --time={days}-00:00:00          # Run time in hh:mm:ss
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --constraint={constraint}
#SBATCH --ntasks-per-node={n_tasks}
#SBATCH --mem-per-cpu={mem_per_cpu}       # Maximum memory required per CPU (in megabytes)
#SBATCH --error={error_log}
#SBATCH --output={output_log}

module load cuda
module load anaconda
conda activate {anaconda_env}

export PYTHONPATH={fcnn_dir}:$PYTHONPATH

{python} {fcnn_dir}/fcnn/scripts/run_trial.py\
 --config_filename {config_filename}\
 --model_filename {model_filename}\
 --training_log_filename {log_filename}\
 --machine_config_filename {machine_config_filename}
""".format(days=days, job_name=job_name, partition=partition, n_gpus=n_gpus, constraint=constraint, n_tasks=n_tasks,
           mem_per_cpu=mem_per_cpu, error_log=error_log, output_log=output_log, anaconda_env=anaconda_env,
           fcnn_dir=fcnn_dir, config_filename=config_filename, model_filename=model_filename, python=python,
           log_filename=training_log_filename,
           machine_config_filename=machine_config_filename)

    slurm_script_filename = os.path.join(output_dir, job_name + ".slurm")
    with open(slurm_script_filename, "w") as opened_file:
        opened_file.write(slurm_script)

    subprocess.call(["sbatch", slurm_script_filename])


def submit_cross_validation_trials(config_filename, n_folds, group="training", **slurm_kwargs):
    config = load_json(config_filename)
    if group not in config:
        from fcnn.utils.filenames import load_subject_ids
        load_subject_ids(config)
    subject_ids = list(config[group])
    shuffle(subject_ids)
    folds = divide_into_folds(subject_ids, n_folds)
    for i, (train, validation) in enumerate(folds):
        config["training"] = train
        config["validation"] = validation
        fold_config_filename = config_filename.replace(".json", "fold{}.json".format(i))
        dump_json(config, fold_config_filename)
        submit_slurm_trial(fold_config_filename, **slurm_kwargs)


def divide_into_folds(x, n_folds):
    folds = list()
    start = 0
    fold_size = len(x)/float(n_folds)
    step = int(fold_size)
    leftover_step = fold_size - step
    leftovers = 0
    for i in range(n_folds):
        stop = (i + 1) * step
        leftovers = leftovers + leftover_step
        while leftovers >= 1 - 1e-4:
            stop = stop + 1
            leftovers = leftovers - 1
        validation = x[start:stop]
        train = x[:start] + x[stop:]
        folds.append([train, validation])
        start = stop
    return folds


def main():
    namespace = parse_args()
    assert os.path.exists(namespace.trial_config)
    slurm_config = load_json(namespace.slurm_config)
    if namespace.n_folds > 1:
        submit_cross_validation_trials(namespace.trial_config, namespace.n_folds, **slurm_config)
    else:
        submit_slurm_trial(namespace.trial_config, output_dir=namespace.output_dir, **slurm_config)


if __name__ == "__main__":
    main()
