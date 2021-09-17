python scripts/plot_correlations.py\
 --correlation_filename /work/aizenberg/dgellis/fCNN/predictions/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_t1t2_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_t1_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_cortexmask_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_wmmaks_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_brainmask_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MSMAll.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_t1t2_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MSMAll.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_t1_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MSMAll.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_cortexmask_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MSMAll.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_wmmask_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MSMAll.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_brainmask_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MSMAll.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MNINonLinear.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_t1t2_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MNINonLinear.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_t1_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MNINonLinear.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_cortexmask_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MNINonLinear.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_wmmask_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MNINonLinear.npy\
 /work/aizenberg/dgellis/fCNN/predictions/v4_brainmask_unet_ALL-TAVOR_2mm_v2_pt_test/overall_correlations_MNINonLinear.npy\
 --output_dir /work/aizenberg/dgellis/fCNN/predictions/figures/test/weighted/struct14\
 --labels DTI+T1+T2-FS T1+T2-FS T1-FS "Cortex Mask-FS" "Subcortical Mask-FS" "Brain Mask-FS"\
 DTI+T1+T2-MSMAll T1+T2-MSMAll T1-MSMAll "Cortex Mask-MSMAll" "Subcortical Mask-MSMAll" "Brain Mask-MSMAll"\
 DTI+T1+T2-MNINonLinear T1+T2-MNINonLinear T1-MNINonLinear "Cortex Mask-MNINonLinear" "Subcortical Mask-MNINonLinear" "Brain Mask-MNINonLinear"
