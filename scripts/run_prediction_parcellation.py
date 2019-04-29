import subprocess
import os
import glob


if __name__ == '__main__':
    label_filename = "/home/aizenberg/dgellis/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
    for gifti_left_filename in glob.glob("/work/aizenberg/dgellis/fCNN/predictions/*.L.*.func.gii"):
        gifti_right_filename = gifti_left_filename.replace(".L.", ".R.")
        cifti_filename = gifti_left_filename.replace(".L.", ".").replace(".func.gii", "dscalar.nii")

        if not os.path.exists(cifti_filename):
            cmd_args = ["/work/aizenberg/dgellis/tools/workbench/bin_rh_linux64/wb_command",
                        "-cifti-create-dense-scalar",
                        cifti_filename,
                        "-left-metric",
                        gifti_left_filename,
                        "-right-metric",
                        gifti_right_filename]
            print(" ".join(cmd_args))
            subprocess.call(cmd_args)

        pscalar_filename = cifti_filename.replace(".dscalar.", ".pscalar.")
        if not os.path.exists(pscalar_filename):
            cmd_args = ["/work/aizenberg/dgellis/tools/workbench/bin_rh_linux64/wb_command",
                        "-cifti-parcellate",
                        cifti_filename,
                        label_filename,
                        "COLUMN",
                        pscalar_filename]
            print(" ".join(cmd_args))
            subprocess.call(cmd_args)
