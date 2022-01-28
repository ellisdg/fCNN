import glob
import subprocess
import os


def main():
    l_roi_wc = "/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/fsaverage_LR32k/{0}.L.atlasroi.32k_fs_LR.shape.gii"
    r_roi_wc = "/work/aizenberg/dgellis/HCP/HCP_1200/{0}/MNINonLinear/fsaverage_LR32k/{0}.R.atlasroi.32k_fs_LR.shape.gii"
    for fn in glob.glob("/work/aizenberg/dgellis/HCP/HCP_1200/*/T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/*_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii"):
        out_file = fn.replace(".midthickness.", ".roi.midthickness.")
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
