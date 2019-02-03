## TASK 7
### Classification of Handwritten Digits using Machine Learning - CCN Based Classifier
- Members :-
    - Adesh Shambu
    - Cheeranjeevi HR
    - Anshul Suresh Bansal

### Architecture 
- Convolution Layer.
    - **img** : 1D vector having 10k * 32 * 32  elements (Zero Padded).
    - **cnnWeight** : 1D (local) vector having 5 * 5 * 32 elements.
    - **cnnBias** : Bias weight for 32 filters. 1D (local) vector having 32 elements.
    - **numberOfImages** : Number of images in the dataset =10k.
    - **numberOfFilters** : number of convolution filters = 32
    - **Output** : 28 * 28 * 32 image will be transferred to MaxPool using channel.
- Maxpool Layer.
    - **Stride** : 2.
    - **Input** : 28 * 28 * 32 Pixels for 1 Image. 10k Images in total.
    - **Output** : 14 * 14 * 32 Pixels for 1 Image, 10K Images in total sent through Channel to FC.
- Fully Connected Layer.
    - **Input** : 14*14*32 pixels for 1 Image. 10K Images in total
    - **Input** : 14*14*32 digit weights(local) for 1 Class. 10 Classes in Total.
    - **Output** : 1 class for each 10K images.

## Implementations
### 1. Simple,Unoptimized version of CNN.
- Kernel Design:-
    - Global Memory used for weights and images.
    - No Pragma Unroll used for the loops.
    - 3 kernels for each layer and data transfer using 1 element(32 bit width) channel.
- Results:-
    - Obtained Accuracy in CPU and FPGA : 97.07%.
    - Kernel Execution time on FPGA : 29.4 seconds.
    - Kernel Execution time in CPU (CC-7) : 71 seconds.
### 2. Kernel Optimizations for the above Kernel.
- Kernel Design :-
    - Local Memory used for weights and images.
    - Pragma Unroll used for the loops.
    - Removed Serial execution regions in the kernel.
- Results:-
    - Obtained Accuracy in CPU and FPGA : 97.07%.
    - Kernel Execution time on FPGA : 3 seconds.
    - Kernel Execution time in CPU (CC-7) : 71 seconds.

### Implementations Results
| Resource  | Utilisation in %(Unoptimized) | Utilisation in %(Optimized)  |
| ---       |---                | ---              |
| Logic Utilization | 33 | 41 |
| ALUT's | 18 | 24 |
| Dedicated Logic Registers | 16 | 19 |
| Memory Blocks | 29 | 56 |
| DSP Blocks | 7 | 10 |

| Parameters | CPU | FPGA(Unoptimized)  |  FPGA(Optimized)  |
| ---       |---                | ---              |----|
| Runtime(sec) | 71 | 29.4 | 3 |

### Performance Models
- Factors constant for each layer
    - Fmax : 174.395 Mhz
    - Total Cycles : 5279128*10^6
    - **Measure Time taken : 3.02711s**
- ##### Convolution Layer
    - Operations:-
        - 5 additions.
        - 5 multiplication.
        - 1 comparision (ReLU).
    - Loops (5 * 28 * 28 * 32 * 10000) :-
        - ADD+MUL = 1254400*10k operations.
        - ReLU = 25088 * 10k operations.
        - Total Operations = 1279488*10k.
    - Unroll Factor=5.
        - 10 operations + 1 comparision.
        - 55 operations per cycle.
- **Measured Total Operations / cycle = 23.56.**
- **Estimated Time : 1.427second.**
- ##### MaxPool Layer
    - Operations:-
        - 3 integer comparison.
    - Loops (14 * 14 * 32 * 10000) :-
        - Total Operations = 75264*10k.
    - No Unroll Factor.
        - 3 operations per cycle.
- **Estimated Time : 1.4385733second.**
- **Measured Total Operations / cycle = 0.36.**
- ##### Fully Connected layer
    - Operations:-
        - 1 addition.
        - 1 multiplication.
        - 1 comparison.
    - Loops (6272 * 10 * 10000):-
        - Total Operations = 188160*10k.
    - Unroll factor = 32 for Add and Mul.
        - 73 Operations.
- **Estimated Time : 0.16second**
- **Measured Total Operations / cycle = 3.56**

### Global Memory Usage
- Total Image Reads: 32 * 32 * 10K * 1 Byte = 10.24MB.
- CNN Filter Weights Read:5 * 5 * 32 * 2 Bytes = 1.6KB.
- CNN Filter Bias Read:32 * 2 Bytes = 64B.
- FC Digit Weights Read: 14 * 14 * 32 * 10 * 2 Bytes = 125.44KB.
- Classified Output Write:10K * 4 Bytes = 40 KB.
- Total Global Memory Reads = 10.367 MB.
- Total Global Memory Writes = 40KB.
- Estimated Global Memory Bandwidth = 7.237MB/s.
- Measured Global Memory Bandwidth  (Profiler):-
    - Conv Kernel : 2 Mem banks 8.2MB/s each = 16.4MB/s.
    - FC Kernel : 2 Mem banks 8.2MB/s each = 16.4MB/s

### Performance Model
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
