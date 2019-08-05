# CustoNN2 : Minutes of the meeting


| Date of Meeting (YYYY/MM/DD)  | (2019/08/05 )  |  
|:--- | :---: |  
| Time  |  09:15 - 11:45 |  
| MOM Prepared by  | Adesh  |  

| Team | |
| --- | :---: |
| Present members | Adesh, Anshul, Alina, Aarthy, Amay, Chiranjeevi, Nikitha | 
| Absentees | Aayush, Suprajith, Rushikesh |

### Notes and Decisions 
##### Project organization updates
-  
##### Research/Individual Task Updates: Task Updates
-  **GoogLeNet optimizations** were divided among the team according to inception modules. Following are the individual updates:
    - Inception 3a and 3b - assigned to Adesh  - Optimizations wrt to II and data dependecies are done, inner loops were unrolled and II were reduced to 1. But Local memory utilizations are high due to replications. 
    - Inception 3c and 4f - assigned to Alina - 3c is optimized with optimizations like unrolling, data dependencies and Loop coalescing. Weights and biases are still in global memory, this needs to be copied to local memory. 4f optimization is not started.
    - Inception 4b and 4c - assinged to Arathy - Data dependencies and II optimizations are done. 
    - Inception 4d and 4e - assinged to Suprajith - Optimizations are completed.
    - Inception 5b and 5c - assinged to Nikitha - II was reduced to 1 and Loop unrolling and Loop coalescing techniques were used. 5c optimization is not started.
**All of these optimized kernels are yet to be tested**.
- **Resnet** : Debugging is in progress. Debug till 3 layers has been completed.
- **Batch support** : Batch support for Googlenet conv and maxpooling are done, concat and padding is yet to be completed.
- Bitstreams for channels implementation of Googlenet was synthesized in `fpgasync` node.
- **Googlenet Baseline** : Googlenet baseline architecture was executed on noctua cluster. But the profile.mon files are being over written. So we need to come up with a solution to generate multiple profile.mon files to measure the metrics.

##### Project decisions
-

### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Multiple profile.mon files | Alina |
| Resnet Debugging | Anshul, Amay, Chiranjeevi |

|Action| Owner|Due Date|
|:--- | :--- | :---: |
| Googlenet Optimizations | Kernel optimization team | 12-08-2019
|  |  |
