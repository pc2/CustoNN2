#!/bin/bash
#SBATCH -N 1
#SBATCH -J inception-4b-4c
#SBATCH -A hpc-lco-kenter
#SBATCH -p short
#SBATCH -t 00:05:00
#SBATCH --mail-type all
#SBATCH --mail-user aayushb@campus.uni-paderborn.de

echo "###############"
echo "*****dummy*****"
echo "###############"

 
aoc  -v -march=emulator -board=p520_max_sg280l GoogleNet_Kernels_channel_1a-3a.cl &&
aoc  -v -march=emulator -board=p520_max_sg280l GoogleNet_Kernels_channel_3b.cl &&

wait

