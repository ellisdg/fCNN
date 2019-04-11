import subprocess
import os


submit=False
overwrite=False


with open("/home/aizenberg/dgellis/HCP/hcp_subjects.txt", "r") as opened_file: 
    for subject_id in opened_file.read().strip().split(","):
        if submit:
            print("Submitting:", subject_id)
            subprocess.call(["sbatch", "/home/aizenberg/dgellis/scripts/run_hcp_cifti_separate.sh", subject_id])
        else:
            hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
            results_dir = "{hcp_dir}/{subject_id}/MNINonLinear/Results".format(hcp_dir=hcp_dir, subject_id=subject_id)
            smoothing_level = 4
            for task_name in ("MOTOR", "LANGUAGE"):                
                task_results_dir = "{results_dir}/tfMRI_{task_name}/tfMRI_{task_name}_hp200_s{smoothing_level}_level2_MSMAll.feat/".format(results_dir=results_dir, 
                                                                                                                                           task_name=task_name, 
                                                                                                                                           smoothing_level=smoothing_level)
                cifti_basename = "{task_results_dir}/{subject_id}_tfMRI_{task_name}_level2_hp200_s{smoothing_level}_MSMAll".format(task_results_dir=task_results_dir,
                                                                                                                                   subject_id=subject_id,
                                                                                                                                   task_name=task_name,
                                                                                                                                   smoothing_level=smoothing_level)
                cifti_filename = "{cifti_basename}.dscalar.nii".format(cifti_basename=cifti_basename)
                left_output = "{cifti_basename}.L.func.gii".format(cifti_basename=cifti_basename)
                right_output = "{cifti_basename}.R.func.gii".format(cifti_basename=cifti_basename)
                if os.path.exists(cifti_filename):
                    if overwrite or not os.path.exists(left_output) or not os.path.exists(right_output):
                        if overwrite:
                            for output in (left_output, right_output):
                                if os.path.exists(output):
                                    os.remove(output)
                        cmd_args = ["/work/aizenberg/dgellis/tools/workbench/bin_rh_linux64/wb_command",
                                    "-cifti-separate",  cifti_filename, "COLUMN",
                                    "-metric", "CORTEX_LEFT", left_output,
                                    "-metric", "CORTEX_RIGHT", right_output]
                        cmd = " ".join(cmd_args)
                        print(cmd)
                        subprocess.call(cmd_args)
                    else:
                        print("Already processed:", cifti_filename)
                else:
                    print("Doesn't exist:", cifti_filename)
