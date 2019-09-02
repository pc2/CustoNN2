# CustoNN2 : Minutes of the meeting
<br/>

| Date of Meeting (YYYY/MM/DD)  | (2019/ 09 / 02 )  |  
|:--- | :---: |  
| Time  |  9:00 - 12:00 |  
| MOM Prepared by  | Suprajith  |  

| Team | |
| --- | :---: |
| Present members | Adesh, Anshul, Amay, Alina, Arathy,Rushikesh,Aayush, Nikhitha, Suprajith| 
| Absentees | Chiranjeevi|

### Notes and Decisions 
##### Project organization updates
- We have to collect as much of results from FPGAs as possible until 14th Sept. After that , Noctua will be down for maintenance.
- 
##### Research/Individual Task Updates:  
- Optimizations : Team has successfully increased resource utilization. But this also is causing synthesis to fail. 
- Googlenet on FPGA : We get correct results only after running multiple times on FPGA. This is because we are not launching kernels which depend on global memory sequentially.



##### Project decisions
 -  Final Documentation and  presentation - team discussed about the skeleton of the document. 
 -  CPU  inference time of GoogleNet : 10ms , Resnet : 20ms. Considering the the time available , synthesis problems etc , it was decided that inference time of around 1s is a good target for us.
 -  Optimization decisions : Because synthesis of our kernels is failing , it was decided to bring down some optimizations to bring down resource and logic utilizations.


### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Scheduling of kernels for GoogLeNet | Suprajith and Rushikesh  | 2019/09/04
| Synthesis fails for optimized kernels  | Aayush | 2019/09/04





|Action| Owner|Due Date|
|:--- | :--- | :---: |
| Modify plugin to launch Hybrid Designs correctly | Suprajith and Rushikesh | 2019/09/04
| Synthesizing with different seeds | Aayush | 2019/09/04
| Resent on 16 FPGAs | Anshul and Nikitha | 2019/09/04
| Optimizations - Resent and Googlenet  | Rest of the team| 2019/09/04
| Doc skeleton  | Amay | 2019/09/04

