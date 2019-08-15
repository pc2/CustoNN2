# CustoNN2 : Minutes of the meeting


| Date of Meeting (YYYY/MM/DD)  | (2019/08/14 )  |  
|:--- | :---: |  
| Time  |  10:00 - 13:00 |  
| MOM Prepared by  | Anshul  |  

| Team | |
| --- | :---: |
| Present members | Aayush, Anshul, Adesh, Chiranjeevi, Alina, Arathy, Nikhitha, Amay, Rushikesh, Suprajith | 
| Absentees |  |

### Notes and Decisions 
##### Project organization updates
-  
- 
##### Research/Individual Task Updates: Task Updates
-  Research about profiler reports in AOCL Best practices guide
-  XML file for route connections in Torus Topology.
-  The team gave an intermediate representation regarding the milestones that have been achieved till date.
    - Feedback for the presentation regarding developement :-
        - Try not to use cl.finish as it causes overhead for kernels with less execution time.
        - Look into launching of kernels using less number of queues
        - Have better understanding of the performance by doing in-dpeth analysis of profiler report
        - `Do not use resource utlisation as sole metric for performance evaluation`
    - Feedback for the presentation regarding organization of slides :-
        - Initial slide be clear about sequence of execution of tools
        - More description about kernel generation and modification required
        - Include details about shortcomings, lessons learned and ways to solve them
        - Give clear idea and description about projection organization.

##### Project decisions
- The team has decided not to pursue the task of Quantization
- The team also decided not to go forward with the third topology, i.e, Inception-V4 as the team focus is primarily on optimizing GoogleNet.


### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Issue 1| Torus Topology has 4 nodes |
| Issue 2 | MPI execution error on Adesh's and Rushikesh's account  |

|Action| Owner|Due Date|
|:--- | :--- | :---: |
| Resnet kernels Optimization | Amay, Arathy, Aayush, Nikitha |
| Plugin integration | Adesh, Rushikesh |
| MPI execution | Chiranjeevi, Alina|