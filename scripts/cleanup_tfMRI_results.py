import glob
import os
import shutil
import subprocess


def run_command(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)


def main():
    for folder in glob.glob("/work/aizenberg/dgellis/HCP/HCP_1200/*/T1w/Results/tfMRI*/tfMRI*s2_level1.feat/StandardVolumeStats"):
        feat_dir = os.path.dirname(folder)
        basename = os.path.basename(feat_dir).replace("_level1.feat", "").replace("_hp", "_{}_hp")
        success = True
        for name in ("cope", "zstat"):
            output_filename = os.path.join(feat_dir, basename.format(name))
            cmd = ["wb_command", "-volume-merge", output_filename]
            if not os.path.exists(output_filename):
                _files = glob.glob(os.path.join(folder, "{}*.nii.gz".format(name)))
                for i in range(len(_files)):
                    _file = os.path.join(folder, "{}{}.nii.gz".format(name, i))
                    cmd.extend(["-volume", _file])
            run_command(cmd)
            if not os.path.exists(output_filename):
                success = False
        if success:
            shutil.rmtree(folder)


if __name__ == "__main__":
    main()
