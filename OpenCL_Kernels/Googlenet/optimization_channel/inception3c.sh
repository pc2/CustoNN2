#!/bin/bash
#SBATCH -N 1
#SBATCH -J inception-3c
#SBATCH -A hpc-lco-k+
#SBATCH -p fpga
#SBATCH -t 10:00:00
#SBATCH --mail-type all
#SBATCH --mail-user aayushb@campus.uni-paderborn.de


make GoogleNet_Kernels_channel_3c