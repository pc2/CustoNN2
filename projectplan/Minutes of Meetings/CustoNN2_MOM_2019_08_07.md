# CustoNN2 : Minutes of the meeting
<br/>

| Date of Meeting (YYYY/MM/DD)  | (2019/ / )  |  
|:--- | :---: |  
| Time  |  09:15 - 11:45 |  
| MOM Prepared by  | Alina  |  

| Team | |
| --- | :---: |
| Present members | Adesh, Anshul, Alina, Aarthy, Amay, Chiranjeevi, Nikitha, Aayush, Rushikesh | 
| Absentees | Suprajith |

### Notes and Decisions 
##### Project organization updates
-  
- 
##### Research/Individual Task Updates: Task Updates
- GoogLeNet optimizations - 3a and 3b are done (including testing), but the local memory utilization is high, but it can be solved in the next iteration of Optimizations. The rest modules are still in progress and have to be tested.
- Resnet - Debugging is complete and ResNet runs successfully on emulation. The problem was with TVM generated ScaleShift and Eltwise kernels. 
- Alina presented the MPI part. The presentation will be uploaded in git.
- Performance modeling is done for GoogLeNet baseline.

##### Project decisions
-
-

### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Issue 1| - |
| Issue 2 | - |

|Action| Owner|Due Date|
|:--- | :--- | :---: |
| MPI integration with ResNet | Chiranjeevi  |
| Splitting ResNet kernels into blocks | Anshul |
| Increasing the batch size for GoogleNet kernels | Amay |