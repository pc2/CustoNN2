# Xilinx ML Suite  
Everything is taken from [here](https://github.com/Xilinx/ml-suite "Xilinx ML Suite")  

The Xilinx Machine Learning (ML) Suite provides users with the tools to develop and deploy Machine Learning applications for Real-time Inference. It provides support for many common machine learning frameworks such as Caffe, Tensorflow, and MXNet.

The ML Suite is composed of three basic parts:

- ### xDNN IP - High Performance general CNN processing engine.  
Xilinx xDNN IP cores are high performance general CNN processing engines. This means they can accept a wide range of CNN networks and models. There are two populat configurations : 28x32 and 56 x 32 .
Each xDNN IP kernel supports popular layers which appear in litrature like Convolution , Deconvolution , Transpose , Max pooling , Avg pooling etc.
Xilinx claims that the implementations of these layers are very energy efficient  (performance/watt).

- ### xfDNN Middleware - Software Library and Tools to Interface with ML Frameworks and optimize them for Real-time Inference.  
xfDNN middleware is a high-performance software library with a well-defined API which acts as a bridge between deep learning frameworks such as Caffe, MxNet, Tensorflow, and xDNN IP running on an FPGA.
xfDNN not only provides simple Python interfaces to connect to high level ML frameworks, but also provides tools for network optimization by fusing layers, optimizing memory dependencies in the network, and pre-scheduling the entire network removing CPU host control bottlenecks.
Once these optimizations are completed per layer, the entire network is optimized for deployment in a "One-Shot" execution flow.
 xfDNN Quantizer enables fast, high-precision calibration to lower precision deployments to INT8 and INT16. 

- ### ML Framework and Open Source Support - Support for high level ML Frameworks and other open source projects.  




## Members
- Amay Churi
- Suprajith SH






