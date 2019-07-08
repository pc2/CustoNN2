# CustoNN2 : Minutes of the meeting
<br/>

| Date of Meeting (YYYY/MM/DD)  | (2019/ 07 / 08 )  |  
|:--- | :---: |  
| Time  |  16:15 - 18:45 |  
| MOM Prepared by  | Suprajith  |  

| Team | |
| --- | :---: |
| Present members | Arathy , Alina , Chiranjeevi , Suprajith , Adesh , Rushikesh , Amay , Aayush | 
| Absentees | Anshul, Nikitha |

### Notes and Decisions 
##### Project organization updates
-  
        
         

##### Research/Individual Task Updates: Task Updates
-   Team was able to find some of the major bugs in Googlenet. Thnings like normalization , channel order (BGR to RGB) etc was discovered during debugging.
-   Replaced all the handwritten kernels of Googlenet with tool generated kernels with appropriate arguments.
-   Comparision of values at intermediate layers of Tensorflow implementation against kernel implementation.
-   Injection of intermediate results inbetween the layers is being done to verify the kernel code.
-   Find out "Layout of weights and pixels in conv layer" - not needed as we already know how it is.
-   Number of Operations in our kernel file has been calculated with the help of Antlr4 tool for performance modelling. It differs from the literature by ~10%

##### Project decisions
 

### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Debugging Network | Suprajith , Rushikesh , Adesh | Wednesday | 
| Performance modelling | Aayush , Amey | Monday | 
| Debugging intermediate inception modules | Chiru , Arathy | Wednesday | 

|Action| Owner|Due Date|
|:--- | :--- | :---: |
| Action1 | - |
| Action2 | - |
