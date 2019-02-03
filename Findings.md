## TASK 7
### Classification of Handwritten Digits using Machine Learning - CCN Based Classifier
- Members :-
    - Adesh Shambu
    - Cheeranjeevi HR
    - Anshul Suresh Bansal

### Architecture 
- Convolution Layer.
    - **Image Input ** : 1D vector having 10k * 32 * 32  elements (Zero Padded).
    - **CNN Weights Input ** : 1D (local) vector having 5 * 5 * 32 elements. This is copied to local memory.
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
