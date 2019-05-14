# Steps to infer the CNN Model using OpenVINO FPGA Plugin on Stratix 10 FPGA

### Connecting the Noctua Cluster
1. Connect to the Noctua Load Balancer
    `ssh fe.noctua.pc2.uni-paderborn.de`
    (This command can be executed from the CC Fe also)

2. Connect to one of the 16 FPGA front nodes  
    `ssh noctua`

3.  Connect to FPGA node (Load Balancer will decide the node) :
  `srun --partition=fpga -A hpc-lco-kenter --constraint=18.1.1_max  --pty bash`
    - for getting the user group run "pc2status" in the fpga front end.
    - If you want to select a particular FPGA, then use `-w ` option in the above command.
4. Load OpenCL,Nallatech and gcc modules
    `module load intelFPGA_pro/18.1.1 nalla_pcie/18.1.1 gcc/6.1.0`

The above procedure has to be followed each time you are executing a bitstream on FPGAs.

### Running Host Codes / Launching OpenVINO 
5. Navigate to DLDT/inference Engine/
    `cd $<DLDT_HOME>`
    `cd inference-engine/bin/intel64/Release/ `
    - (optional) //Set a env variable to point the Intermediate representation directory:
`export IR='/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation'`
    - `/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation` directory is the repository for OpenVINO intermediate representation files (with bin and xml).

6.Execute the test_plugin using FPGA Plugin Inference.
 We had written a user application called "test_plugin" to launch the SimpleCNN model using OpenVINO FPGA Plugin. The command to execute the application is :  
 `./test_plugin -m $IR/lenet_iter_10000.xml  -i $IR/one.png > out_noctua.txt`
    - parameters for test plugin:
        - `-m` : path to the CNN Model name (IR XML file)
        - `-i` : Directory where the images are stored. If we use wildcard like zero_*.png , all the png images matching the wildcard will be used for the inference.
        - Only .PNG images are supported for OpenVINO.
