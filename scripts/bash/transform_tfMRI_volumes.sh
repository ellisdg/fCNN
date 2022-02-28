export PATH=$WORK/tools/workbench/bin_rh_linux64:$PATH

HCPDIR=${WORK}/HCP/HCP_1200
SUB=${1}
SUBDIR=${HCPDIR}/${SUB}
TFMRIVOL=${SUBDIR}/T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/${SUB}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.nii.gz
XFM=${SUBDIR}/MNINonLinear/xfms/acpc_dc2standard.nii.gz
REF=${SUBDIR}/MNINonLinear/T1w_restore.2.nii.gz
REF2=${SUBDIR}/T1w/T1w_acpc_dc_restore_brain.nii.gz
OUTDIR=${SUBDIR}/MNINonLinear/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat

mkdir -p "${OUTDIR}"

OUTVOL=${OUTDIR}/${SUB}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.nii.gz

wb_command -volume-warpfield-resample\
 "${TFMRIVOL}"\
 "${XFM}"\
 "${REF}"\
 CUBIC\
 "${OUTVOL}"\
 -fnirt "${REF2}"
