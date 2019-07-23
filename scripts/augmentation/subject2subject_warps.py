import sys
import itertools
import os
import subprocess
import time

from fcnn.utils.utils import load_json


def main(subjects_filename, hcp_dir, output_dir, relative_path, bash_script, sub_limit=200, after_limit_wait=0.1):
    subjects_dict = load_json(subjects_filename)
    subjects = subjects_dict["training"]
    for i, (subject1, subject2) in enumerate(itertools.combinations(subjects, 2)):
        if i > sub_limit:
            time.sleep(after_limit_wait)
        subject1 = str(subject1)
        subject2 = str(subject2)
        warp1 = os.path.join(hcp_dir, subject1, "MNINonLinear", "xfms", "acpc_dc2standard.nii.gz")
        warp2 = os.path.join(hcp_dir, subject2, "MNINonLinear", "xfms", "acpc_dc2standard.nii.gz")
        ref_filename1 = os.path.join(hcp_dir, subject1, "T1w", "T1w_acpc_dc_restore_1.25.nii.gz")
        ref_filename2 = os.path.join(hcp_dir, subject2, "T1w", "T1w_acpc_dc_restore_1.25.nii.gz")
        comp_warp_filename1 = os.path.join(output_dir, "{0}_2_{1}_T1w_1.25.nii.gz".format(subject1, subject2))
        comp_warp_filename2 = os.path.join(output_dir, "{1}_2_{0}_T1w_1.25.nii.gz".format(subject1, subject2))
        moving1 = os.path.join(hcp_dir, subject1, relative_path)
        moving2 = os.path.join(hcp_dir, subject2, relative_path)
        output_dir1 = os.path.join(hcp_dir, subject1, os.path.dirname(relative_path), "augmented")
        output_dir2 = os.path.join(hcp_dir, subject2, os.path.dirname(relative_path), "augmented")
        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)
        out1 = os.path.join(output_dir1, "_".join((subject2, os.path.basename(relative_path))))
        out2 = os.path.join(output_dir1, "_".join((subject1, os.path.basename(relative_path))))
        if not os.path.exists(out1) and not os.path.exists(out2):
            cmd = ["sbatch", bash_script, warp1, warp2, ref_filename1, ref_filename2, comp_warp_filename1,
                   comp_warp_filename2, moving1, moving2, out1, out2]
            print(" ".join(cmd))
            subprocess.call(cmd)


if __name__ == "__main__":
    subjects_filename = sys.argv[1]
    hcp_dir = sys.argv[2]
    output_dir = sys.argv[3]
    relative_path = sys.argv[4]
    bash_script = sys.argv[5]
    main(subjects_filename=subjects_filename, hcp_dir=hcp_dir, output_dir=output_dir, relative_path=relative_path,
         bash_script=bash_script)
