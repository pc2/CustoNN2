# Intel OPENVINO
- OPENVINO is a toolkit that supports the deployment of pre-trained deep neural network models on different hardware platforms such as CPU, GPU, FPGA etc. 
- It supports a number of deep learning frameworks such as TensorFlow, Caffe, PyTorch, MX net etc. 

## Components
- Model Optimizer: Converts a pre-trained model into a common intermediate representation (independent of the framework) consisting of a .xml file and a .bin file.
	1. .xml file stores the computational graph topology
	2. .bin file stores the weights for each layer
- Inference Engine: It provides a C++ API to deploy the network on the required hardware platform.  
- Both the components are open source. Link for the Github repository:
https://github.com/opencv/dldt

## Findings
- Inference Engine source code for FGPA processing wasn't found in the repository.
- Supports only ARRIA 10, we need check for possibilities to run on Stratix 10 FPGAs
- When installed locally,
	- Case1:
dataset: ImageNet
Model: Inception V4 model
Accuracy: Great
Average processing time on CPU per Image: 100+ms
Average processing time on CPU for a batch of 9images: 750+ms

	- Case2:
 dataset: ImageNet
Model: Resnet 101
Accuracy: poor
Average processing time on CPU per Image: 100+ms
Average processing time on CPU for a batch of 7images: 588+ms

## Members
- Nikhitha Shiwaswamy
- Rushikesh Nagle





