# CustoNN2 : Minutes of the meeting
| Date of Meeting (YYYY/MM/DD)  | (2019/09/04 )  |  
|:--- | :---: |  
| Time  |  09:15 - 12:00 |  
| MOM Prepared by  | Adesh  |  

| Team | |
| --- | :---: |
| Present members | Alina, Suprajith, Nikitha, Adesh | 
| Absentees | Amay, Chiranjeevi, Rushikesh, Aayush, Arathy, Anshul |

### Notes and Decisions 
##### Project organization updates
-  Noctua clusters going on maintanance from 16th Sept. We have to finish our inference and results collection before this deadline.
- Tobias will not be available next week ( 09 -13 Sept )
##### Research/Individual Task Updates: Task Updates
-  Googlenet bitstreams: 
    -  Inception 3a, 3b and 4b synthesis failed in the routing stage. 
    -  Inception 4c,4d and 4e synthesis was successful but we noticed that the Fmax was too low in the designs.
    -  Tobias suggested these synthesis failure might be due to global memory accesses. So we might have to use channels between the kernels.
    -  `-global-ring -duplicate-ring` needs to be used while synthesizing to achieve better performance in 19.1 BSP.
- Hybrid channel design for Googlenet was executed on FPGAs and inference time was 6.18 seconds. Speed up of 12.88 was achieved compared to baseline.
- Accuracy metric: We can compare results and likelihood with CPU results for multiple images.
- Batching : Lower priority compared to optimizations, but batching will increase the throughput.
- Resnet : The topology was split into 16 units. It will be optimized to increase number of operations per cycle.
##### Project decisions
-
-

### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Googlenet synthesis | Optimizations team |
| Resnet-50 | splitting the topology into 16 units & synthesis |

|Action| Owner|Due Date|
|:--- | :--- | :---: |
| - | - |
