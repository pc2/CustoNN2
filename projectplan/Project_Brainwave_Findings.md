
# Microsoft Project Brainwave
- Microsoft's Project Brainwave is the deep learning platform for real time Artificial Intelligence applications.
- Project Brainwave is the first of its kind to prove the value of FPGAs for Cloud Computing, It has specialized hardware and algorithms for High throughput and Ultra-low latency accelerated serving of DNNs.
- It has following applications:
  - Conversational agents.
  - Computer Vision(CNNs).
  - Natural Language Processing.
  - Intelligent Search Engines.
- The main purpose of this Project is to enable users utilize  Accelerated Hardware System for DNNs without any knowledge of Hardware Architecture or HDL.

## Datacenter Architecure
- Brainwave is built on Microsoft hyperscale datacenter architecture.
- Each Catapult servers has Dual Core Xeon CPUs with PCIe attached FPGA.
- Multiple FPGAs can be allocated for a application as a single shared Hardware Microservice.
- Each FPGAs are in-line between Server's NIC Card and Top-of-Rack switch enabling point to point connectivity at low latency.
- This kind of architecture helps in scalable workloads and better load Balancing between CPUs and FPGAs

## Project Brainwave Architecture
Brainwave is built using three layers:  
- Pretrained DNNs Models:  
  - A tool flow that converts pre trained DNN Models into a deployment Package.  
  - Several Popular Machine Learning Frameworks(Tensorflow/Caffe/CNTK) can be used to develop the DNN Model.
  - The model is further drilled down to Graphical Intermediate Representations(IR).
  - This graph is partitioned into sub graphs using greedy algorithms and these sub graphs are assigned to different CPUs or FPGAs.

- Scalable DNN Hardware Microservice:
    - This is the Hardware side of the Brainwave
    - It consists of several FPGAs connected to each other using Network switches.
    - The subgraphs from IR are mapped to FPGAs and When the memory resource of the FPGA is exhausted, the other subgraphs are mapped to the next free FPGA. This way the Hardware is scalable.
- Brainwave Soft NPU(Neural Processing Unit):
  - The Programmable Brainwave NPU hosted on the FPGAs
  - This is the Brain of the Project which enables Real Time Applications.
  - It has the mega-SIMD vector processor architecture.
  - NPUs are capable of processing Single DNN requests at low batch with high utilization.
  - Conventially, the number of requests are batched for Better Utilization of a Resource(like GPUs) but NPUs are capable enough to serve high utilization with one request also.
  - Maximizes Hardware Efficeny at low batch sizes.

## Microsoft Azure and Project Brainwave
- Microsoft Azure provides End-to-End Data Science Platform - Azure ML Design Suite.
- Azure helps in data preparation and Model Training.
- It provides Hardware(FPGAs) as a service.
- The type of deployment can be either Cloud or Edge.
- 1 Azure Box contains 24 CPU Cores + 4 Arria 10 FPGAs
- Provides Limited and Standard DNNs : ResNet 50,ResNet 152,VGG-16,SSD-VGG, and DenseNet-121

## Deploying trained DNN models to FPGAs in the Azure cloud
###Prerequisites
  - Azure subscription
  - Azure Machine Learning service workspace
  - Azure Machine Learning SDK for Python
  - tensorflow version<=1.10 
  - Python 3.6
  - Anaconda  

###Design
  - Dataset : ImageNet
  - CNN Used : ResNet50
  - Input :JPEG Images
 
###Steps
    - Preprocess the Image and produce a Tensor
    - Use ResNet50 as a featurizer
    - Classify the output of ResNet50 into top 5 classes 
    - Create a service defintion (this contains a pipeline of stages that are required to deploy the model on FPGA)
    - Create and Use a service 
    - Clean Up the service

## Results
- Brainwave successfully exploits FPGAs on a datacenter-scale fabric for real-time serving of state-of-the-art DNNs.
-Designing a scalable, end-to-end system architecture for deep learning is as critical as optimizing for single chip performance
- Today, Project Brainwave serves DNNs in real time for production services such as Bing Intelligent Search.

## References
- https://www.microsoft.com/en-us/research/uploads/prod/2018/03/mi0218_Chung-2018Mar25.pdf
- https://www.microsoft.com/en-us/research/project/project-brainwave/
- https://github.com/Azure/aml-real-time-ai
- https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-fpga-web-service

## Members:
- Adesh Shambhu (adeshs@main.uni-paderborn.de)
- Chiranjeevi Hogalli Revanna (chiruhr@main.uni-paderborn.de)
