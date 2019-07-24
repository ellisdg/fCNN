#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=dti_aug
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out
warp1S=${1}
warp2S=${2}
warpS1=${3}
warpS2=${4}
ref1=${5}
ref2=${6}
compwarp12=${7}
compwarp21=${8}
in1=${9}
in2=${10}
fnout1=${11}
fnout2=${12}
export PATH=/work/aizenberg/dgellis/tools/fsl/bin:/work/aizenberg/dgellis/tools/workbench/bin_rh_linux64:${PATH}
export LD_LIBRARY_PATH=/home/aizenberg/dgellis/.conda/envs/fcnn-1.12/lib:${LD_LIBRARY_PATH}
convertwarp --warp1=${warp1S} --warp2=${warpS2} --ref=${ref2} -o ${compwarp12}
convertwarp --warp1=${warp2S} --warp2=${warpS1} --ref=${ref1} -o ${compwarp21}
wb_command -volume-warpfield-resample ${in1} ${compwarp12} ${ref2} TRILINEAR ${fnout1} -fnirt ${ref2}
wb_command -volume-warpfield-resample ${in2} ${compwarp21} ${ref1} TRILINEAR ${fnout2} -fnirt ${ref1}
rm ${compwarp12}
rm ${compwarp21}

