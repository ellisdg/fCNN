import argparse
import os
from slurm.utils import submit_slurm_gpu_process


def parse_args():
    from fcnn.scripts.run_unet_inference import format_parser as format_prediction_parser

    parser = argparse.ArgumentParser()
    format_prediction_parser(parser, sub_command=False)
    parser.add_argument("--slurm_options", help="Add slurm options in quotations in order with no quotations.",
                        nargs="*")
    parser.add_argument("--slurm_filename", default="/work/aizenberg/dgellis/{job}_{model}.slurm")
    parser.add_argument("--error_log", default="/work/aizenberg/dgellis/job.{job}_{model}_{placeholder}.err")
    parser.add_argument("--output_log", default="/work/aizenberg/dgellis/job.{job}_{model}_{placeholder}.out")
    parser.add_argument("--anaconda_env", default="pytorch-1.5")
    parser.add_argument("--local", action="store_true", default=False)
    return vars(parser.parse_args())


def format_process(**kwargs):
    process = "python fcnn/scripts/run_unet_inference.py"
    for key, value in kwargs.items():
        if value:
            if type(value) == list:
                value = " ".join(value)
            process = "{process} --{key} {value}".format(process=process, key=key, value=value)
    return process


def main():
    kwargs = parse_args()
    if kwargs["slurm_options"]:
        slurm_options = ["--" + option for option in kwargs.pop("slurm_options")]
    else:
        slurm_options = []
    anaconda_env = kwargs.pop("anaconda_env")
    job_name = "predict_{}_{}".format(os.path.basename(kwargs["config_filename"]).split(".")[0].replace("_config", ""),
                                      kwargs["group"])
    model_name = os.path.basename(kwargs["model_filename"]).split(".")[0]
    slurm_filename = kwargs.pop("slurm_filename").format(job=job_name, model=model_name)
    error_log = kwargs.pop("error_log").format(job=job_name, model=model_name, placeholder="%J")
    output_log = kwargs.pop("output_log").format(job=job_name, model=model_name, placeholder="%J")
    local = kwargs.pop("local")
    process = format_process(**kwargs)
    submit_slurm_gpu_process(process=process, slurm_script_filename=slurm_filename, slurm_options=slurm_options,
                             job_name=job_name, anaconda_env=anaconda_env, local=local, error_log=error_log,
                             output_log=output_log)


if __name__ == "__main__":
    main()
