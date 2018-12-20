# Task7 CNN based Classifier
In this task, we implemented a convolutional neural network based classifier on MNIST dataset. The Network has 3 layers : Convolutional Layer,MaxPool Layer and Fully Connected Layer.

### Team InSight Members:
- Adesh Shambhu
- Anshul Suresh Bansal
- Chiranjeevi Hongalli Revanna

### Task Parameters:
- Image Size : 28*28
- Number of Images : 10,000
- Zero Padding : 2
- Convolution Filter mask : 5*5
- Number of Convultion filter : 32
- Maxpool stride : 2
- Number of Digit Classes : 10

### Implementation 1 : Simple,Unoptimized version of CNN
- Here we implemented the unoptimized version of CNN.  
- **Kernel Design :**
 - Global Memory used for weights and images
 - No Pragma Unroll used for the loops
 - 3 kernels and data transfer using 1 element(32 bit width) channel.  
- Obtained Accuracy in CPU and FPGA : 97%
- Kernel Execution time on FPGA : 29.4 seconds
- Kernel Execution time in CPU (CC-7) : 71 seconds
