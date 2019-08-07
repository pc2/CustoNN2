# Executing ResNet in emulation mode -- Only for testing the CNN model
Please follow the instructions below to execute ResNet CNN on 1 FPGA emulation mode.
### Create the single bitsream for ResNet.
All the OpenCl kernels for ResNet are placed in resnet_updated.cl file. To create the bitstream of the model, we have created a makefile with a target to generate aocx under a specific directory. 
1. Initialize Stratix 10 19.1 toolkit  
    `source /opt/intelFPGA_pro/19.1/init_env_bittware_pcie.sh`
2. Emulate the device  
    `export CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=p520_max_sg280l`
3. Generate aocx in emulation mode
    - Navigate to <PG_HOME>/OpenCL_Kernels/ResNet_50/kernels
      `cd $<PG_HOME>/OpenCL_Kernels/ResNet_50/kernels`
    - Make command:  
    `make resnet`  
    This will place the bitsteam in `custonn2/designs/resnet_emulation` directory.  
### Build the OpenVINO noctua plugin
 OpenVINO noctua plugin is built to integrate IR from Model Optimizer with OpenCL Kernels generated from TVM and launch these kernels on 1 FPGA in emulation mode.
We have developed the plugin for this testing branch `noctua_plugin_resnet` of `dldt` project. Please switch to this branch if you are in different branch.
 1. Navigate to build directory of OpenVINO inference engine:  
    `cd $<dldt>/inference-engine/build`
2. Build the plugin  
    `make -j16`  
    If you are building the Inference Engine for the first time , please run Cmake command given this documentation: https://git.uni-paderborn.de/cs-hit/pg-custonn2-2018-3rd-party/dldt/blob/noctua_plugin_develop/inference-engine/build_instructions_pc2.md.
 3. After building the plugin, navigate to bin directory of inference engine:  
    `cd $<dldt>/inference-engine/bin/intel64/DEBUG` If you have built the project in Debug mode   
    `cd $<dldt>/inference-engine/bin/intel64/Release` if the project is built in Release mode.
 4. Execute the model
    `./test_plugin -m /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation/ResNet/resnet_frozen.xml -i /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation/pepper.png`  
    - Test Plugin is the user application for executing the plugin
    - `-m` is the path for IR XML 
    - `-i` is the path of the Image
 