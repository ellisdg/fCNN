import os
import glob

from slurm.submit_predictions import format_process
from slurm.utils import submit_slurm_gpu_process


def main():
    machine_config_filename = "/home/aizenberg/dgellis/fCNN/data/hcc_v100_2gpu_32gb_config.json"
    config_filename_tmp = "/home/aizenberg/dgellis/fCNN/data/brats_config_{fold}.json"
    model_filename_tmp = "/work/aizenberg/dgellis/MICCAI_BraTS2020/models/model_brats_{fold}.h5"
    prediction_dir_tmp = "/work/aizenberg/dgellis/MICCAI_BraTS2020/predictions/{group}/brats_config_{fold}_no_seg"
    error_log_tmp = "/work/aizenberg/dgellis/MICCAI_BraTS2020/predictions/job.{group}_{fold}_{placeholder}.err"
    output_log_tmp = "/work/aizenberg/dgellis/MICCAI_BraTS2020/predictions/job.{group}_{fold}_{placeholder}.out"
    group = "test"
    for fold in range(10):
        config_filename = config_filename_tmp.format(fold=fold)
        model_filename = model_filename_tmp.format(fold=fold)
        error_log = error_log_tmp.format(group=group, fold=fold, placeholder="%J")
        output_log = output_log_tmp.format(group=group, fold=fold, placeholder="%J")
        anaconda_env = "pytorch-1.5"
        job_name = "{}_predict".format(fold)
        slurm_filename = "/work/aizenberg/dgellis/MICCAI_BraTS2020/predictions/{job}.slurm".format(job=job_name)
        processes = list()
        processes.append(format_process(group=group, replace="Training Testing",
                                        output_directory=prediction_dir_tmp.format(group=group, fold=fold),
                                        config_filename=config_filename, model_filename=model_filename,
                                        machine_config_filename=machine_config_filename,
                                        alternate_prediction_func="predictions_with_permutations",
                                        output_template="BraTS20_Testing_{subject}.nii.gz"))
        process = "\n\n".join(processes)
        submit_slurm_gpu_process(process=process, slurm_script_filename=slurm_filename,
                                 job_name=job_name, anaconda_env=anaconda_env, error_log=error_log,
                                 output_log=output_log, days=2)


if __name__ == "__main__":
    main()
