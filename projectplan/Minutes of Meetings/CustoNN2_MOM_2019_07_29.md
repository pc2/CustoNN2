# CustoNN2 : Minutes of the meeting
<br/>

| Date of Meeting (YYYY/MM/DD)  | (2019/ 07 / 29 )  |  
|:--- | :---: |  
| Time  |  9:00 - 12:00 |  
| MOM Prepared by  | Suprajith  |  

| Team | |
| --- | :---: |
| Present members | Adesh, Anshul, Amay, Alina, Arathy, Chiranjeevi,Aayush, Nikhitha, Suprajith| 
| Absentees | Rushikesh|

### Notes and Decisions 
##### Project organization updates
- Amay was chosen as the TL for the next 3 weeks
- It was decided that we will give a presentation on 14th Aug about our progress. The main objective is to get feedback from Dr.Kenter and to act on them before we enter the last phase of the project in the month of Sept.

##### Research/Individual Task Updates:  
-  Problem with modeling our performncae model. We do not know how to model global memory transfer rate
-  The problem of "Assertion failure and crash" - was it related to system maintenance ? We do not know.
-  The team working on channels (IO and internal) has tested and verfied the functioning of the same
-  MPI - Successfully executed inceptions on one node (2 FPGAs)
-  Resnet plugin : Tree structure rectified to include shortcut conv kernels , which weren`t appearing in the tree earlier
-  Resnet kernels : Resnet kernel file has been modified as per requirements of plug in team
-  Dynamic IO channels have been introduced and are being tested


##### Project decisions
 -  Batch mode : It  is helpful for increasing throughput of our system. More research needs to be done in this area.
 -  We are supposed to show results of Googlenet at different optimization points. No need to do the same with Resnet - just baseline and final optimized results are enough.
 -  We decided to modify the scope of the project plan a little. 

### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Testing IO Dynamic Channels| Suprajith  | 2019/31/07
| MPI dev and test  | Alina | 2019/31/07
| Resnet plugin  | Chiranjeevi, Aayush | 2019/31/07
| Resnet kernels  | Amay | 2019/31/07


|Action| Owner|Due Date|
|:--- | :--- | :---: |
| Action1 | - |
| Action2 | - |
