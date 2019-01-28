# Background Areas of Interest and Starting Points

## General Background

### Popular Datasets, CNN Architectures, Trained Models
- Summarize knowledge from Stanford lecture and tutorial
- Extend with information from given and further references - also articles that focus on different topic can contain interesting details about the used benchmarks

### Evaluation Metrics and Design Tradeoffs
- How do people evaluate their CNN designs and hardware implementations?
- What evaluation metrics should we focus on in this project, what values can we target?

## Specific Material

### Procedures

- Assign each source to 2 group members.
- Mark material as taken and specify deadline to finish preparation. - Read and research material independently, get in touch with other group member when aspects are unclear.
- After deadline discuss and answer the following questions.
- Document findings in gitlab and also clearly mark open points that require further understanding.
- Prepare a mini presentation (2-4 slides) as discussion basis for next group meeting.

#### Questions to answer

- What is the novelty, special selling point of this work?
- How does this work connect to the rest of the field with regard to the aspects mentioned in general background?
- Which of these aspects could we like to use, reproduce or exceed in our work?

### Ecosystems and Frameworks

#### Intel OpenVINO (Nikhitha and Rushikesh)
- [Intel OpenVINO Website](https://software.intel.com/en-us/openvino-toolkit)
- [Dell Success Story using OpenVINO](https://www.nextplatform.com/2018/10/15/where-the-fpga-hits-the-server-road-for-inference-acceleration/)

#### Xilinx (Amay and Suprajith)
- [Overview of Xilinx Activities](http://www.ispd.cc/slides/2018/s2_3.pdf)
- [Xilinx ML Suite](https://github.com/Xilinx/ml-suite)
- [Deephi - Embedded Counterpart to Xilinx xfDNN](https://www.xilinx.com/publications/events/developer-forum/2018-frankfurt/xilinx-machine-learning-strategies-with-deephi-tech.pdf)
- [Omnitek - Success Story using Xilinx HW with a Custom Architecture](https://www.nextplatform.com/2018/10/01/boosting-the-clock-for-high-performance-fpga-inference/)

#### tvm (Alina and Arathy)
- [TVM Website](https://tvm.ai/)
- [TVM Github Page](https://github.com/dmlc/tvm/)
- [VTA Announcement](https://tvm.ai/2018/07/12/vta-release-announcement.html)

#### Microsoft (Adesh and Chiranjeevi)
- [Microsoft Project Brainwave Homepage](https://www.microsoft.com/en-us/research/project/project-brainwave/)
- [Microsoft Project Brainwave Background](https://www.microsoft.com/en-us/research/blog/microsoft-unveils-project-brainwave/)
- [Using Project Brainwave on Azure Cloud](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-accelerate-with-fpgas)

#### Surveys (Anshul and Ayush)
- [Survey of Academic Publications on Frameworks](https://arxiv.org/pdf/1803.05900.pdf)
- [Very Good Survey on Approximation for Custom Hardware](https://arxiv.org/abs/1901.06955)

### Research

#### Reusable Architecture Design 
- [Xilinx GEMX](https://github.com/xilinx/gemx)

#### Custom Precision
- [Xilinx FINN](https://dl.acm.org/citation.cfm?id=3021744)
- [Xilinx FINN-R](https://dl.acm.org/citation.cfm?id=3242897)
- [Xilinx FINN-L](https://ieeexplore.ieee.org/abstract/document/8533474)

#### Sparsity
- [Pruning and Retraining for Sparsity: "Learning both weights and connections for efficient neural networks"](https://dl.acm.org/citation.cfm?id=2969366)
- [Accelerator Architecture for Execution of Pruned Neural Networks: "EIE: efficient inference engine on compressed deep neural network"](https://dl.acm.org/citation.cfm?id=3001163)
- ["ESE: Efficient Speech Recognition Engine with Sparse LSTM on FPGA"](https://dl.acm.org/citation.cfm?id=3021745)

#### GANs
- ["FlexiGAN: An End-to-End Solution for FPGA Acceleration of Generative Adversarial Networks"](https://www.cc.gatech.edu/~hadi/doc/paper/2018-fccm-flexigan.pdf)

# Project Directions

- What elements from the state-of-the art do we want to support?
- Which frameworks or tools would we want to build upon?
- Which cool and novel features do we want to work on?
- How can we exploit the unique scaling capabilities at PC2 with 32 FPGAs and custom point-to-point channels between several FPGAs?
	- Which essential limitations can that architecture overcome?
	- In which workloads / scenarios do these limitations show up?