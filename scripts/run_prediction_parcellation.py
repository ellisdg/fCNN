import subprocess
import os
import glob


if __name__ == '__main__':
    label_filename = "/home/aizenberg/dgellis/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
    for cifti_filename in glob.glob("/work/aizenberg/dgellis/fCNN/predictions/*.dscalar.nii"):
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
