# Steps to infer the CNN Model using OpenVINO FPGA Plugin on Stratix 10 FPGA

### Connecting the Noctua Cluster
1. Connect to the Noctua Load Balancer  
    `ssh fe.noctua.pc2.uni-paderborn.de`  
    (This command can be executed from the CC Fe also)

2. Connect to Noctua FPGA front nodes  
    `ssh noctua`

3.  Connect to one of the 16 FPGA nodes (Load Balancer will decide the node) :  
  `srun --partition=fpga -A hpc-lco-kenter --constraint=18.1.1_max  --pty bash`
    - for getting the user group run "pc2status" in the fpga front end.
    - If you want to select a particular FPGA, then use `-w ` option in the above command.
4. Load OpenCL,Nallatech and gcc modules  
    `module load intelFPGA_pro/18.1.1 nalla_pcie/18.1.1 gcc/6.1.0`

    The above procedure has to be followed each time you are executing a bitstream on FPGAs.

### Running Host Codes / Launching OpenVINO 
5. Navigate to DLDT/inference_engine/  
    `cd $<DLDT_HOME>`
    `cd inference-engine/bin/intel64/Release/ `
    - (optional) Set a env variable to point the Intermediate representation directory:  
    `export IR='/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation'`
    - `/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation` directory is the repository for OpenVINO intermediate representation files (with bin and xml).
    - `/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs` is the FPGA bitstream repo.
6. Execute the test_plugin using FPGA Plugin Inference.  
 We have written a user application called "test_plugin" to launch the SimpleCNN model using OpenVINO FPGA Plugin. The command to execute the application is :  
 `./test_plugin -m $IR/lenet_iter_10000.xml  -i $IR/one.png > out_noctua.txt`
    - parameters for test plugin:
        - `-m` : path to the CNN Model name (IR XML file)
        - `-i` : Directory where the images are stored. If we use wildcard like zero_*.png , all the png images matching the wildcard will be used for the inference.
        - Only .PNG images are supported for OpenVINO.
        - NOTE: The model name and the bitstream(aocx) name should be same.



# Executing GoogLeNet in emulation mode -- Only for testing the CNN model
Please follow the instructions below to execute GoogLeNet CNN on 1 FPGA emulation mode.
### Create the single bitsream for googlenet.
All the OpenCl kernels for googlenet are placed in GoogleNet_kernels.cl file. To create the bitstream of the model, we have created a makefile with a target to generate aocx under a specific directory. 
1. Initialize Stratix 10 19.1 toolkit  
    `source /opt/intelFPGA_pro/19.1/init_env_bittware_pcie.sh`
2. Emulate the device  
    `export CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=p520_max_sg280l`
3. Generate aocx in emulation mode
    - Navigate to <PG_HOME>/OpenCL_Kernels/kernels/nnvm
      `cd $<PG_HOME>/OpenCL_Kernels/kernels/nnvm`
    - Make command:  
    `make googlenet`  
    This will place the bitsteam in `custonn2/designs/inception_modified_nnvm` directory.  
### Build the OpenVINO noctua plugin
 OpenVINO noctua plugin is built to integrate IR from Model Optimizer with OpenCL Kernels generated from TVM and launch these kernels on 1 FPGA in emulation mode.
We have developed the plugin for this testing branch `noctua_plugin_new_googlenet` of `dldt` project. Please switch to this branch if you are in different branch.
 1. Navigate to build directory of OpenVINO inference engine:  
    `cd $<dldt>/inference-engine/build`
2. Build the plugin  
    `make -j16`  
    If you are building the Inference Engine for the first time , please run Cmake command given this documentation: https://git.uni-paderborn.de/cs-hit/pg-custonn2-2018-3rd-party/dldt/blob/noctua_plugin_develop/inference-engine/build_instructions_pc2.md.
 3. After building the plugin, navigate to bin directory of inference engine:  
    `cd $<dldt>/inference-engine/bin/intel64/DEBUG` If you have built the project in Debug mode   
    `cd $<dldt>/inference-engine/bin/intel64/Release` if the project is built in Release mode.
 4. Execute the model
    `./test_plugin -m /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation/GoogLeNet/frozen_quant.xml -i /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation/pepper.png`  
    - Test Plugin is the user application for executing the plugin
    - `-m` is the path for IR XML 
    - `-i` is the path of the Image
 
 
 
