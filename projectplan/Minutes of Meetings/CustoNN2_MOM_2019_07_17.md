# CustoNN2 : Minutes of the meeting
<br/>

| Date of Meeting (YYYY/MM/DD)  | (2019/17/07 )  |  
|:--- | :---: |  
| Time  |  09:00 - 12:00 |  
| MOM Prepared by  | Chiranjeevi H R  |  

| Team | |
| --- | :---: |
| Present members | Adesh, Arathy, Rushikesh, Suprajith,Aayush, Anshul, Amey, Chiranjeevi, Nikitha |
| Absentees | Alina |

### Notes and Decisions
##### Project organization updates
-  
-

##### Research/Individual Task Updates: Task Updates
- Adesh informed that IO Channels between the kernels have been implemented but it has not been tested. Tobias advised to use feeder kernels and launch them without arguments in a while loop.
- Suprajith and Aayush updated that they were able to run inception 3b on FPGA and showed the findings from the profiler.
- Tobias informed the team to use the code generation with TVM and hardware emulation as a baseline the intial results are a comparision point for future optimizations.
- Rushikesh updated that he is working on integrating launching of kernels with MPI.
-

##### Project decisions
- The team informed Tobias that Optmizations on Googlenet kernels will start from implementing channels in the kernels to overcome the Global memory barrier. And later, work on reducing the initiation intervals of the kernels.
- The team informed Tobias that they will parellely start working on Resnet 50 implementation.
- Tobias informed the team to use CC front end instead of noctua and read through the Slurm documentation thoroughly before running anything on noctua.

### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Issue 1| - |
| Issue 2| - |

|Action| Owner|Due Date|
|:--- | :--- | :---: |
| Testing IO Channels| Adesh  | 2019/17/07
| Generate IR for Resnet50  | Amey and Rushikesh | 2019/17/07
| Implement and test channels in inception 3b  | Arathy | 2019/17/07
| Generate Kernels for Resnet50 using TVM  | Amey | 2019/17/07
