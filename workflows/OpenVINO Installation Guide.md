# OpenVINO Installation guide on CC Frontend
Download the Intel® Distribution of OpenVINO™ 2019 R1 toolkit package file from Intel® Distribution of OpenVINO™ toolkit for Linux* with FPGA Support [`here`](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux-fpga). Select the Intel® Distribution of OpenVINO™ toolkit for Linux with FPGA Support package from the dropdown menu.

Open a command prompt terminal window.
Change directories to where you downloaded the Intel Distribution of OpenVINO toolkit for Linux* with FPGA Support package file.
If you downloaded the package file to the current user's `Downloads` directory: 

    cd ~/Downloads/
    
By default, the file is saved as `l_openvino_toolkit_fpga_p_<version>.tgz`.
Unpack the .tgz file: 
    
    tar -xvzf l_openvino_toolkit_fpga_p_<version>.tgz 
The files are unpacked to the `l_openvino_toolkit_fpga_p_<version>` directory.
Go to the `l_openvino_toolkit_fpga_p_<version>` directory:

    cd l_openvino_toolkit_fpga_p_<version>
If you have a previous version of the Intel Distribution of OpenVINO toolkit installed, rename or delete these two directories:
    - `/home/<user>/inference_engine_samples`
    - `/home/<user>/openvino_models`
    
#### Installation Notes:

Choose an installation option and run the related script as root.
You can use either a GUI installation wizard or command-line instructions (CLI).
The following information applies to CLI and will be helpful to your installation where you will be presented with the same choices and tasks.
Choose your installation option:

Option 1: GUI Installation Wizard:
    
    sudo ./install_GUI.sh
        
Option 2: Command-Line Instructions:
    
    sudo ./install.sh
    
Follow the instructions on your screen. 

#### Configure the Model Optimizer

The Model Optimizer is a Python*-based command line tool for importing trained models from popular deep learning frameworks such as Caffe*, TensorFlow*, Apache MXNet*, ONNX* and Kaldi*.

The Model Optimizer is a key component of the Intel Distribution of OpenVINO toolkit. You cannot perform inference on your trained model without running the model through the Model Optimizer. When you run a pre-trained model through the Model Optimizer, your output is an Intermediate Representation (IR) of the network. The Intermediate Representation is a pair of files that describe the whole model:

`.xml`: Describes the network topology
`.bin`: Contains the weights and biases binary data

##### Model Optimizer Configuration Steps

You can choose to either configure all supported frameworks at once OR configure one framework at a time. Choose the option that best suits your needs. If you see error messages, make sure you installed all dependencies.
If you installed the Intel® Distribution of OpenVINO™ to the non-default install directory, replace `/opt/intel` with the directory in which you installed the software.

**Option 1: Configure all supported frameworks at the same time**

Go to the Model Optimizer prerequisites directory:
    
    cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites 
    
Run the script to configure the Model Optimizer for Caffe and TensorFlow:
    
    sudo ./install_prerequisites.sh

**Option 2: Configure each framework separately**

Configure individual frameworks separately ONLY if you did not select Option 1 above.
Go to the Model Optimizer prerequisites directory:

    cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
Run the script for your model framework. You can run more than one script:

**For Caffe:**

    sudo ./install_prerequisites_caffe.sh

**For TensorFlow:**

    sudo ./install_prerequisites_tf.sh
    
The Model Optimizer is configured for two frameworks.

OpenVINO does not support bitstreams that are compatible with FPGA on PC2 Infrastructure. Hence it requires development of custom plugin and bitstreams for respective boards. 

Detailed installation guide can be found here:
https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux_fpga.html


