# Task7 CNN based Classifier
In this task, we implemented a convolutional neural network based classifier on MNIST dataset.  
The Network has 3 layers : Convolutional Layer, MaxPool Layer and Fully Connected Layer. Each Layer is implemented in an OpenCL kernel.

### Team InSight Members:
- Adesh Shambhu
- Anshul Suresh Bansal
- Chiranjeevi Hongalli Revanna

### Architecture
- Convolution Layer.
    - **Image Input** : 1D vector having 10k * 32 * 32  elements (Zero Padded).
    - **CNN Weights Input** : 1D (local) vector having 5 * 5 * 32 elements. This is copied to local memory.
    - **CNN Bias Input** : Bias weight for 32 filters. 1D (local) vector having 32 elements. . This is copied to local memory.
    - **Output** : 28 * 28 * 32 image will be transferred to MaxPool using channel.
- Maxpool Layer.
    - **Stride** : 2.
    - **Input** : 28 * 28 * 32 Pixels for 1 Image. 10k Images in total.
    - **Output** : 14 * 14 * 32 Pixels for 1 Image, 10K Images in total sent through Channel to FC.
- Fully Connected Layer.
    - **Input** : 14*14*32 pixels for 1 Image. 10K Images in total
    - **Input** : 14*14*32 digit weights(local) for 1 Class. 10 Classes in Total.
    - **Output** : 1 class for each 10K images.
- Two Channels to send Intermediate results(1 Row of Data in our case) to next Kernel/Layer.

### Implementation 1 : Simple, Unoptimized version of the CNN
- Here we implemented the unoptimized version of CNN.  
- **Kernel Design :**
    - Global Memory used for weights and images
    - No Pragma Unroll used for the loops
    - 3 kernels for each layer and data transfer using 1 element(32 bit width) channel.  
- Obtained Accuracy in CPU and FPGA : 97.07%
- Kernel Execution time on FPGA : 29.4 seconds
- Kernel Execution time in CPU (CC-7) : 71 seconds

### Implementation 2 : Optimized version of the CNN
- Kernel Design :-
    - Local Memory used for weights and images.
    - Pragma Unroll used for the loops.
    - Removed Serial execution regions in the kernel.
    - Widened Channel Implementation : Sends 1 row of intermediate result to next Kernel/Layer.
- Results:-
    - Obtained Accuracy in CPU and FPGA : 97.07%.
    - Kernel Execution time on FPGA : **157.15ms**.
    - Kernel Execution time in CPU (CC-7) : 71 seconds.

### Estimated Performance Models
    - Factors constant for each layer
        - Fmax : 179.05 Mhz
        - Total Cycles : 28.137*10^6
        - Measured Time : 157.15ms
    - ##### Convolution Layer
        - Operations:-
            - 5 additions.
            - 5 multiplication.
            - 1 comparision (ReLU).
        - Loops (5 * 28 * 28 * 32 * 10000) :-
            - ADD+MUL = 1254400*10k operations.
            - ReLU = 25088 * 10k operations.
            - Total Operations = 1279488*10k.
        - Unroll Factor=5*28.
            - 1568 operations per cycle.
        - Estimated Time : 45.5ms
    - ##### MaxPool Layer
        - Operations:-
            - 3 integer comparison.
        - Loops (14 * 14 * 32 * 10000) :-
            - Total Operations = 75264*10k.
        - Unroll Factor=14.
            - 3*14 = 42 operations per cycle.
    - Estimated Time : 1ms.

    - ##### Fully Connected layer
        - Operations:-
            - 1 addition.
            - 1 multiplication.
            - 1 comparison.
        - Loops (6272 * 10 * 10000):-
            - Total Operations = 188160*10k.
        - Unroll factor = 32 for Add and Mul.
            - 73 Operations.
    - Estimated Time : 143.9ms

### Measured Performance Model Results:
| Conv.Layer | Time in (ms) | Ops/Cycle  |
| ---       |---                | ---              |
| Estimated | 45.5 | 1568 |
|Measured | 157.15 | 454.7 |

| MaxPool Layer |  Time in (ms) | Ops/Cycle  |
| ---       |---                | ---              |
| Estimated | 100 | 42 |
|Measured | 157.15 | 26.75 |

| FC Layer  |  Time in (ms) | Ops/Cycle  |
| ---       |---                | ---              |
| Estimated | 143.9 | 73 |
| Measured | 157.15 | 66.87 |

- Fmax : 179.05MHz.
- Classification time per Image : 15.71us.
- FC Layer was our slowest kernel.

### How to Execute the Project:
We have archived the necessary project files under the project design directory :  
- /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/Task7Group_InSight/  
Project archive File : Task7_GroupInSight_archive.tar  

The .tar file contains the compiled host code,.cl kernel code, synthesized kernel code, quartus report and a buildme.txt(with commands to synthesize the kernel and run profiler).  
To execute the project, extract the archive and execute the object file *host_prog* under the directory *host_src*.
