import subprocess
import argparse
import os
from fcnn.utils.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser(description='Use wb_command to compute stats over a group.')
    parser.add_argument('--subjects_filename', help='JSON filename containing a dictionary of the subjects.')
    parser.add_argument('--group', help='Group name (dictionary key) of the subjects to use.')
    parser.add_argument('--template',
                        help='Template of the cifti filename to compute statistics over. '
                             '"{subject}" should be used as a placeholder for the subject ID.')
    parser.add_argument('--output_filename', help='output filename with "{group}" and "{operation}" used as '
                                                  'placeholders.')
    parser.add_argument('--overwrite', action="store_true", default=False)
    return vars(parser.parse_args())


def run_command(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)


def compute_mean(output_filename, subjects, cifti_template, overwrite=False):
    if not os.path.exists(output_filename) and not overwrite:
        cmd = ["wb_command", "-cifti-average", output_filename]
        for subject in subjects:
            cifti_filename = cifti_template.format(subject=subject)
            if os.path.exists(cifti_filename):
                cmd.extend(["-cifti", cifti_filename])
        run_command(cmd)


def compute_std(output_filename, mean_filename, subjects, cifti_template, overwrite=False):
    if not os.path.exists(output_filename) and not overwrite:
        n = 0
        var_args = ["-var", "mean", mean_filename]
        squared_deviation_expressions = list()
        for subject in subjects:
            cifti_filename = cifti_template.format(subject=subject)
            var = "S" + subject
            if os.path.exists(cifti_filename):
                var_args.extend(["-var", var, cifti_filename])
                n += 1
                squared_deviation_expressions.append("(({} - mean)^2)".format(var))
        sum_of_squared_deviations_exression = "+".join(squared_deviation_expressions)
        expression = "sqrt(({0})/({1}-1))".format(sum_of_squared_deviations_exression, n)
        cmd = ["wb_command", "-cifti-math", expression, output_filename] + var_args
        run_command(cmd)


def compute_cv(output_filename, mean_filename, std_filename, overwrite=False):
    if not os.path.exists(output_filename) and not overwrite:
        expression = "std/mean"
        cmd = ["wb_command", "-cifti-math", expression, output_filename,
               "-var", "std", std_filename,
               "-var", "mean", mean_filename]
        run_command(cmd)


def main():
    args = parse_args()
    mean_filename = args['output_filename'].format(group=args['group'], operation="mean")
    subjects = load_json(args['subjects_filename'])[args['group']]
    compute_mean(output_filename=mean_filename, subjects=subjects, cifti_template=args['template'],
                 overwrite=args["overwrite"])
    std_filename = args['output_filename'].format(group=args['group'], operation="std")
    compute_std(output_filename=std_filename, mean_filename=mean_filename, subjects=subjects,
                cifti_template=args['template'], overwrite=args['overwrite'])
    cv_filename = args['output_filename'].format(group=args['group'], operation="cv")
    compute_cv(output_filename=cv_filename, mean_filename=mean_filename, std_filename=std_filename,
               overwrite=args['overwrite'])


if __name__ == "__main__":
    main()
