#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --job-name=WB18LS
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000       # Maximum memory required per CPU (in megabytes)
#SBATCH --error=/work/aizenberg/dgellis/fCNN/logs/job.%J.err
#SBATCH --output=/work/aizenberg/dgellis/fCNN/logs/job.%J.out
warp1=${1}
warp2=${2}
ref1=${3}
ref2=${4}
compwarp12=${5}
compwarp21=${6}
in1=${7}
in2=${8}
fnout1=${9}
fnout2=${10}
convertwarp --warp1=${warp1} --warp2=${warp2} --ref=${ref2} -o ${compwarp12}
convertwarp --warp1=${warp2} --warp2=${warp1} --ref=${ref1} -o ${compwarp21}
wb_command -volume-warpfield-resample ${in1} ${compwarp12} ${ref2} TRILINEAR ${fnout1} -fnirt ${ref2}
wb_command -volume-warpfield-resample ${in2} ${compwarp21} ${ref1} TRILINEAR ${fnout2} -fnirt ${ref1}
rm ${compwarp12}
rm ${compwarp21}

