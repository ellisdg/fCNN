export PATH=$WORK/tools/workbench/bin_rh_linux64:$PATH

HCPDIR=${WORK}/HCP/HCP_1200
SUB=${1}
SUBDIR=${HCPDIR}/${SUB}
BASENAME="struct14_normalized.nii.gz"
TFMRIVOL=${SUBDIR}/T1w/${BASENAME}
XFM=${SUBDIR}/MNINonLinear/xfms/acpc_dc2standard.nii.gz
REF=${SUBDIR}/MNINonLinear/T1w_restore.nii.gz
REF2=${SUBDIR}/T1w/T1w_acpc_dc_restore_brain.nii.gz
OUTDIR=${SUBDIR}/MNINonLinear/

mkdir -p "${OUTDIR}"

OUTVOL=${OUTDIR}/${BASENAME}

wb_command -volume-warpfield-resample\
 "${TFMRIVOL}"\
 "${XFM}"\
 "${REF}"\
 CUBIC\
 "${OUTVOL}"\
 -fnirt "${REF2}"
