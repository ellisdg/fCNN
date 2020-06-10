#!/usr/bin/env bash

fCNNDir=$HOME/fCNN
LabelsFN=$fCNNDir/data/labels/ALL-TAVOR_name-file.txt
python $fCNNDir/scripts/plot_compared_correlation_models.py \
$WORK/fCNN/predictions/v4_t1_unet_ALL-TAVOR_2mm_v2_pt_test/correlations.npy,\
$WORK/fCNN/predictions/v4_t1t2_unet_ALL-TAVOR_2mm_v2_pt_test/correlations.npy,\
$WORK/fCNN/predictions/v4_struct6_unet_ALL-TAVOR_2mm_v2_pt_test/correlations.npy,\
$WORK/fCNN/predictions/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test/correlations.npy,\
$WORK/fCNN/predictions/v4_t1_unet_ALL-TAVOR_2mm_v2_pt_test/correlations_MSMAll.npy,\
$WORK/fCNN/predictions/v4_t1t2_unet_ALL-TAVOR_2mm_v2_pt_test/correlations_MSMAll.npy,\
$WORK/fCNN/predictions/v4_struct6_unet_ALL-TAVOR_2mm_v2_pt_test/correlations_MSMAll.npy,\
$WORK/fCNN/predictions/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test/correlations_MSMAll.npy,\
$WORK/fCNN/predictions/v4_t1_unet_ALL-TAVOR_2mm_v2_pt_test/MNINonLinear/MNINonLinearcorrelations.npy,\
$WORK/fCNN/predictions/v4_t1t2_unet_ALL-TAVOR_2mm_v2_pt_test/MNINonLinear/MNINonLinearcorrelations.npy,\
$WORK/fCNN/predictions/v4_struct6_unet_ALL-TAVOR_2mm_v2_pt_test/MNINonLinear/MNINonLinearcorrelations.npy,\
$WORK/fCNN/predictions/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test/MNINonLinear/MNINonLinearcorrelations.npy\
 $WORK/fCNN/predictions/figures/test/weighted/struct14\
 T1-FS,T1+T2-FS,T1+T2+DTI4-FS,T1+T2+DTI12-FS,T1-MSMAll,T1+T2-MSMAll,T1+T2+DTI4-MSMAll,T1+T2+DTI12-MSMAll,T1-MNINonLinear,T1+T2-MNINonLinear,T1+T2+DTI4-MNINonLinear,T1+T2+DTI12-MNINonLinear
