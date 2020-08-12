import argparse
import os
from fcnn.scripts.run_unet_inference import format_parser as format_prediction_parser
from slurm.utils import submit_slurm_gpu_process


def parse_args():
    parser = argparse.ArgumentParser()
    format_prediction_parser(parser, sub_command=False)
    parser.add_argument("--slurm_options", help="Add slurm optoins in quotations in order with no quotations.",
                        nargs="*")
    parser.add_argument("--slurm_filename", default="/work/aizenberg/dgellis/predict_{job}_{model}.slurm")
    parser.add_argument("--anaconda_env", default="fcnn-1.12")
    parser.add_argument("--local", action="store_true", default=False)
    return vars(parser.parse_args())


def format_process(**kwargs):
    process = "python fcnn/scripts/run_unet_inference.py"
    for key, value in kwargs.items():
        if value:
            process = "{process} --{key} {value}".format(process=process, key=key, value=value)
    return process


def main():
    kwargs = parse_args()
    slurm_options = ["--" + option for option in kwargs.pop("slurm_options")]
    anaconda_env = kwargs.pop("anaconda_env")
    job_name = "predict_{}_{}".format(os.path.basename(kwargs["config_filename"]).split(".")[0].replace("_config", ""),
                                      kwargs["group"])
    slurm_filename = kwargs.pop("slurm_filename").format(job=job_name,
                                                         model=os.path.basename(kwargs["model_filename"]).split(".")[0])
    local = kwargs.pop("local")
    process = format_process(**kwargs)
    submit_slurm_gpu_process(process=process, slurm_script_filename=slurm_filename, slurm_options=slurm_options,
                             job_name=job_name, anaconda_env=anaconda_env, local=local)


if __name__ == "__main__":
    main()
