# CustoNN2 : Minutes of the meeting
<br/>

| Date of Meeting (YYYY/MM/DD)  | (2019/07/10)  |  
|:--- | :---: |  
| Time  |  16:00 - 19:00 |  
| MOM Prepared by  | Adesh  |  

| Team | |
| --- | :---: |
| Present members | Arathy, Chiranjeevi , Suprajith , Adesh , Rushikesh , Amay | 
| Absentees | Anshul, Nikitha, Aayush, Alina |

### Notes and Decisions 
##### Project organization updates
- 
##### Research/Individual Task Updates: Task Updates
- Team was still getting wrong results in the Googlenet network 
- The intermediate results obtained from googlenet kernels were presented to Tobias and we found out that outputs from the first conv layer were wrong.
- The memory layout of the kernels were discussed. 
- Tobias suggested to come up with test scenarios with sample data to test the functional correctness of the conv/maxpool/concat kernels.
- Performance modelling : Gave updates on number of operations of inception modules as per Antlr4 tool.
##### Project decisions
- Meeting time during the semester break will be Mondays(9:15-12:15) in the lab and on Wednesdays(9:15-12:15) in the regular meeting room.

### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Debugging Network - Compare the ouputs of each layer with TF and collect the results | Suprajith, Adesh, Rushikesh | |
| Performance modelling  | Aayush ||
| Get IR from TVM after optimizations | Amay ||

|Action| Owner|Due Date|
|:--- | :--- | :---: |
| - | - |
