# Build Inference Engine Project on CC Frontend.
## Software Requirements:
- cmake 3.9 or higher
- gcc 4.8 or higher

### Cloning the DLDT repository
We need to clone the open source Deep Learning Deployment Toolkit repository.
- execute `git@git.uni-paderborn.de:cs-hit/pg-custonn2-2018-3rd-party/dldt.git` to clone the DLDT repository.
- Once the DLDT repository is cloned, we need to clone a sub project `ade` into the project. Navigate to dldt dir and execute the command `git submodule init` followed by `git submodule update --recursive`.
### Build Steps:
- Navigate to <inference Engine directory.> `cd <dldt>/inference-engine`
- Create a build directory `mkdir build`
- Inference Engine uses a CMake based build system. In the `build` directory run the cmake to create the makefile
- cmake command : `cmake3 -DCMAKE_BUILD_TYPE=Release -DENABLE_CLDNN=OFF -DENABLE_GNA=OFF ..`  
The above command is by disabling GPU and GNA Plugin, If you want to enable those, please set the arguments to ON
- Once the cmake command is executed succesfully, run the `make` command to build the DLDT project.  
command : `make -j16`
- To switch on/off the CPU and GPU plugins, use cmake options -DENABLE_MKL_DNN=ON/OFF and -DENABLE_CLDNN=ON/OFF.


### Adding plugin library in your application
- For CMake projects, set an environment variable InferenceEngine_DIR: ` export InferenceEngine_DIR=/path/to/dldt/inference-engine/build/`
- Then you can find Inference Engine by find_package in applciation cmake:  
`find_package(InferenceEngine)`  
`include_directories(${InferenceEngine_INCLUDE_DIRS})`  
`target_link_libraries(${PROJECT_NAME} ${InferenceEngine_LIBRARIES} dl)`  

### Running the plugin on fpgas:
Please refer to this work instruction:
https://git.uni-paderborn.de/cs-hit/pg-custonn2-2018/blob/master/workflows/workflow_OpenVINO_Inference.md

# Build Inference Engine Project on Noctua Nodes
The documentation for building the plugin on Noctua nodes are given in the README files of kernels: https://git.uni-paderborn.de/cs-hit/pg-custonn2-2018/tree/tvm/OpenCL_Kernels/Googlenet/baseline_architecture