All the synthesis job on the Noctua Cluster should be done on **FPGA nodes** and not on Noctua Frontend.

1. Connect to noctua cluster via `ssh fe.noctua.pc2.uni-paderborn.de`
2. Type `ssh noctua` to connect to the noctua frontend
3. Load the toolchain for the BSP `module load intelFPGA_pro/19.0.1 nalla_pcie/19.1.0`
4. To connect to one of the fpga nodes `srun --partition=fpga -A hpc-lco-kenter --constraint=19.1.0 --pty bash` type 

Now to submit the jobs we will use an shell script which will be executed using a command line tool **sbatch**
For e.g. `sbatch myscript.sh` 

__Contents of myscript.sh__

#!/bin/bash
- #SBATCH -N 1
- #SBATCH -J cp2k_water_128
- #SBATCH -A hpc-prf-ldft
- #SBATCH -p fpga
- #SBATCH -t 10:00:00
- #SBATCH --mail-type all
- #SBATCH --mail-user test@example.com

#run your application here

- N 1 = Tells that it is a one node job
- J = Job name given by the user 
- A = Account name. In our case it is `hpc-lco-k+`
- p = It is the partition that we will use for our synthesis processes. The partition which we will use is `fpga`
- t = It is the maximum time limit for our job execution 
- --mail = Send email notification to the user regarding events
- run = here we specify the commands which we want to execute. So for e.g. we can create a Makefile for synthesis process and then specify the target here 
- 
