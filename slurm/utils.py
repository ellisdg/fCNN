import subprocess
import os

slurm_header = """#!/bin/sh
#SBATCH --time={days}-00:00:00          # Run time in hh:mm:ss
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --constraint={constraint}
#SBATCH --ntasks-per-node={n_tasks}
#SBATCH --mem-per-cpu={mem_per_cpu}       # Maximum memory required per CPU (in megabytes)
#SBATCH --error={error_log}
#SBATCH --output={output_log}
"""

slurm_setup = """
module load cuda
module load anaconda
conda activate {anaconda_env}

export PYTHONPATH={fcnn_dir}:$PYTHONPATH
"""


def format_slurm_script(process, header=slurm_header, setup=slurm_setup, **kwargs):
    script = (header + setup).format(**kwargs)
    return script + process


def submit_slurm_process(process, slurm_script_filename, slurm_options=None, local=False, **kwargs):
    slurm_script = format_slurm_script(process, **kwargs)
    with open(slurm_script_filename, "w") as opened_file:
        opened_file.write(slurm_script)
    if local:
        cmd = ["bash"]
    else:
        cmd = ["sbatch"]
        if slurm_options:
            cmd.extend(slurm_options)
    cmd.append(slurm_script_filename)
    print(" ".join(cmd))
    subprocess.call(cmd)


def submit_slurm_gpu_process(process, slurm_script_filename, slurm_options=None, job_name="fcnn", partition="gpu",
                             n_gpus=2, constraint="gpu_v100", days=7, n_tasks=40, mem_per_cpu=4000,
                             error_log="./job.fcnn_%J.err", output_log="./job.fcnn_%J.out", anaconda_env="fcnn-1.12",
                             fcnn_dir=os.path.abspath(os.path.dirname(os.path.dirname(__file__))), local=False):
    return submit_slurm_process(process, slurm_script_filename, slurm_options, job_name=job_name, partition=partition,
                                days=days, n_gpus=n_gpus, constraint=constraint, n_tasks=n_tasks,
                                mem_per_cpu=mem_per_cpu, error_log=error_log, output_log=output_log,
                                anaconda_env=anaconda_env, fcnn_dir=fcnn_dir, local=local)
