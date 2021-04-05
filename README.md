## CustoNN2 - Customizing Neural Networks on FPGA

 - This repository contains all the necessary files for the deployment of the deep neural networks like GoogleNet [1] and ResNet-50 [2] on on multiple Intel Stratix 10 FPGAs using an custom built FPGA plugin.
 - Using a combination of two machine learning frameworks - TVM and Intel OpenVINO, CNNs are tested on the ImageNet dataset. 
 - TVM (Tensor Virtual Machine) is used for OpenCL code generation of the CNN topologies.
 - The OpenVINO Computer Vision Toolkit is used as an inference engine for running the generated code on FPGAs.


## Folder structure

<pre>
CustoNN2  
 --- OpenCl_Kernels 	    			//OpenCl Kernels for GoogleNet and ResNet-50.  
     --- Googlenet          			// GoogleNet Kernels. 
	 --- baseline_architecture      	// Baseline Kernels without any optimization.
	 --- global_optimized       		// Increased DSP usage and global memory optimizations with the help of loop unrolling etc.
	 --- hybrid               		// Data transfer optimized with the help of internal channels, I/O channels and global memory.
     --- Resnet_50          			// ResNet-50 Kernels.
	 --- baseline             		// Baseline Kernels without any optimization.
	 --- optimized_v1               	// Global memory optimizations with the help of loop unrolling  and loop pipelining.
	 --- units_opv1                 	// Divind into 16 units of the model and global memory optimizations with the help of loop unrolling etc.
  --- dldt         				// Sub-directory for FPGA Plugin. dldt - Deep Learning Deployment Toolkit.
      --- inference_engine
	  --- src
	      --- noctua_plugin
		  --- fpga_plugin.cpp 		//FPGA Plugin in C++ to deploy neural networks.
</pre>	

The READMEs in the sub-directories give extensive information on the kernels and the steps needed to execute the FPGA Plugin.

## References

<a id="1">[1]</a> Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott E. Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich:  **Going  Deeper  with Convolutions**.

<a id="1">[2]</a> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun:  **Deep  Residual  Learning  for  Image  Recognition**.

## License
The repository is licensed under [Apache 2.0](https://github.com/pc2/CustoNN2/blob/main/LICENSE) License.
