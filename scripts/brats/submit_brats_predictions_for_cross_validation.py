import os
import glob

from slurm.submit_predictions import format_process
from slurm.utils import submit_slurm_gpu_process


def main():
    machine_config_filename = "/home/aizenberg/dgellis/fCNN/data/hcc_v100_2gpu_32gb_config.json"
    config_filename_tmp = "/home/aizenberg/dgellis/fCNN/data/brats_config_{fold}.json"
    model_filename_tmp = "/work/aizenberg/dgellis/MICCAI_BraTS2020/models/model_brats_{fold}.h5"
    prediction_dir_tmp = "/work/aizenberg/dgellis/MICCAI_BraTS2020/predictions/{group}/brats_config_{fold}_no_seg"
    error_log_tmp = "/work/aizenberg/dgellis/MICCAI_BraTS2020/predictions/job.{fold}_{placeholder}.err"
    output_log_tmp = "/work/aizenberg/dgellis/MICCAI_BraTS2020/predictions/job.{fold}_{placeholder}.out"
    for training_job_fn in glob.glob("/work/aizenberg/dgellis/MICCAI_BraTS2020/models/job.brats_fold*_*.err"):
        _, fold, jobid, = os.path.basename(training_job_fn).replace(".err", "").split("_")
        config_filename = config_filename_tmp.format(fold=fold)
        model_filename = model_filename_tmp.format(fold=fold)
        error_log = error_log_tmp.format(fold=fold, placeholder="%J")
        output_log = output_log_tmp.format(fold=fold, placeholder="%J")
        slurm_options = ["--dependency=afterany:{jobid}".format(jobid=jobid)]
        anaconda_env = "fcnn-1.12"
        job_name = "{}_predict".format(fold)
        slurm_filename = "/work/aizenberg/dgellis/MICCAI_BraTS2020/predictions/{job}.slurm".format(job=job_name)
        processes = list()
        processes.append(format_process(group="test_validation", replace="Training Validation",
                                        output_directory=prediction_dir_tmp.format(group="validation", fold=fold),
                                        config_filename=config_filename, model_filename=model_filename,
                                        machine_config_filename=machine_config_filename,
                                        alternate_prediction_func="predictions_with_permutations",
                                        output_template="BraTS20_Validation_{subject}.nii.gz"))
        processes.append(format_process(group="validation",
                                        output_directory=prediction_dir_tmp.format(group="training", fold=fold),
                                        config_filename=config_filename, model_filename=model_filename,
                                        machine_config_filename=machine_config_filename,
                                        alternate_prediction_func="predictions_with_permutations",
                                        output_template="BraTS20_Validation_{subject}.nii.gz"))
        processes.append(format_process(group="test_validation", replace="Training Validation",
                                        output_directory=prediction_dir_tmp.format(group="validation",
                                                                                   fold=fold + "_best"),
                                        config_filename=config_filename,
                                        model_filename=model_filename.replace(".h5", "_best.h5"),
                                        machine_config_filename=machine_config_filename,
                                        alternate_prediction_func="predictions_with_permutations",
                                        output_template="BraTS20_Validation_{subject}.nii.gz"))
        processes.append(format_process(group="validation",
                                        output_directory=prediction_dir_tmp.format(group="training",
                                                                                   fold=fold + "_best"),
                                        config_filename=config_filename,
                                        model_filename=model_filename.replace(".h5", "_best.h5"),
                                        machine_config_filename=machine_config_filename,
                                        alternate_prediction_func="predictions_with_permutations",
                                        output_template="BraTS20_Validation_{subject}.nii.gz"))
        process = "\n\n".join(processes)
        submit_slurm_gpu_process(process=process, slurm_script_filename=slurm_filename, slurm_options=slurm_options,
                                 job_name=job_name, anaconda_env=anaconda_env, error_log=error_log,
                                 output_log=output_log)


if __name__ == "__main__":
    main()
