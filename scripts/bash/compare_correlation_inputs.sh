#!/usr/bin/env bash

fCNNDir=$HOME/fCNN
LabelsFN=$fCNNDir/data/labels/ALL-TAVOR_name-file.txt
python $fCNNDir/scripts/plot_compared_correlation_models.py \
$WORK/fCNN/predictions/v4_t1_unet_ALL-TAVOR_2mm_v2_pt_test/correlations.npy,\
$WORK/fCNN/predictions/v4_t1t2_unet_ALL-TAVOR_2mm_v2_pt_test/correlations.npy,\
$WORK/fCNN/predictions/v4_struct6_unet_ALL-TAVOR_2mm_v2_pt_test/correlations.npy,\
$WORK/fCNN/predictions/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_test/correlations.npy\
 $LabelsFN,$LabelsFN,$LabelsFN,$LabelsFN\
 $WORK/fCNN/predictions/figures/test/weighted/struct14\
 T1,T1+T2,T1+T2+DTI4,T1+T2+DTI12
