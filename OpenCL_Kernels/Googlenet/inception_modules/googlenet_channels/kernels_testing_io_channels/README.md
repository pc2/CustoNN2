# Work instructions for testing GoogLeNet with channels
### Googlenet with only IO channels:
- Plugin code for testing kernels with only IO channels is in `io_channels_testing_branch` branch
- GoogLeNet kernels with IO channels are saved in `pg-custonn2-2018/OpenCL_Kernels/Googlenet/inception_modules/googlenet_channels/kernels_testing_io_channels`
    - GoogleNet_Kernels_IO.cl (Kernels till Inception 4c)
    - GoogleNet_Kernels_IO_1.cl (Inception 4d to 5c and output layers)
- Use the make command to compile the kernels : `make io_emulation`
- Build the plugin
- initialize stratix 19.1 BSP : `source /opt/intelFPGA_pro/19.1/init_env_bittware_pcie.sh`
- emulate 2 devices : `export CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=2`
- Run the plugin :
    ./test_plugin -m /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation/GoogLeNet/frozen_quant.xml -i /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation/pepper.png 
### Googlenet with IO and internal channels integrated:
- Plugin code for testing kernels with only IO channels is in `channels_testing` branch
- GoogLeNet kernels with IO channels are saved in `pg-custonn2-2018/OpenCL_Kernels/Googlenet/inception_modules/googlenet_channels/kernels_testing_io_channels`
    - GoogleNet_Kernels_channels.cl (Kernels till Inception 4c)
    - GoogleNet_Kernels_channels_1.cl (Inception 4d to 5c and output layers)
- Use the make command to compile the kernels : `make emulate_kernels`
- Build the plugin
- initialize stratix 19.1 BSP : `source /opt/intelFPGA_pro/19.1/init_env_bittware_pcie.sh`
- emulate 2 devices : `export CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=2`
- Run the plugin :
    ./test_plugin -m /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation/GoogLeNet/frozen_quant.xml -i /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/intermediate_representation/pepper.png 
