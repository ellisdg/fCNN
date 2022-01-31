import glob
import subprocess
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wildcard",
                        help="Wildcard for glob to get the files that to which the roi will be applied",
                        default="/work/aizenberg/dgellis/HCP/HCP_1200/*/T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/*_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii")
    parser.add_argument("--surface_name", default="midthickness")
    parser.add_argument("--replace", default=".roi.{surface}.",
                        help="How to label the output files. '.{surface}.' will be replaced with the string here. "
                             "The default is to place '.roi.' before the surface name in the file. But this can "
                             "cause issues for other scripts. So for prediction directories, it is best to place the"
                             "roi after the surface name like this '.{surface}.roi.'")
    return parser.parse_args()


def main():
    namespace = parse_args()
    l_roi_wc = "/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/fsaverage_LR32k/{0}.L.atlasroi.32k_fs_LR.shape.gii"
    r_roi_wc = "/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/fsaverage_LR32k/{0}.R.atlasroi.32k_fs_LR.shape.gii"
    for fn in glob.glob(namespace.wildcard.format(surface=namespace.surface_name)):
        out_file = fn.replace(".{surface}.".format(surface=namespace.surface_name),
                              namespace.replace(surface=namespace.surface_name))
        if not os.path.exists(out_file):
            subject = fn.split("/")[6]
            cmd = ["wb_command",
                   "-cifti-restrict-dense-map",
                   fn, "COLUMN", out_file,
                   "-left-roi", l_roi_wc.format(subject),
                   "-right-roi", r_roi_wc.format(subject)]
            print(" ".join(cmd))
            subprocess.call(cmd)


if __name__ == "__main__":
    main()
