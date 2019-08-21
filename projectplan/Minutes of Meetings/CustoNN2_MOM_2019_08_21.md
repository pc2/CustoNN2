# CustoNN2 : Minutes of the meeting
<br/>

| Date of Meeting (YYYY/MM/DD)  | (2019/08/21)  |  
|:--- | :---: |  
| Time  |  09:15 - 12:00 |  
| MOM Prepared by  | Chiranjeevi H R |  

| Team | |
| --- | :---: |
| Present members | All |
| Absentees | - |

### Notes and Decisions
##### Project organization updates
- Tobias informed the team to keep in mind the maintenance scheduled for for CC and Noctua front end and collect the results well early.
##### Research/Individual Task Updates: Task Updates
-  Alina presented the team about her optimization design for convolution kernels which have yielded DSP usage of ~73% and II of 1 (On emulation). Tobias informed the team to apply the same design for remaining inception modules.
-  ResNet optimization team updated that First round of optimizations have been completed and will be queued for synthesis.
-  Amey informed Tobias regarding the shell script he wrote to collect different profile.mon and Tobias suggested to improve the script to synchronize with how different nodes are launched and writing the profile.mon file.
- Team updated Tobias about the issue team is facing w.r.to running GoogleNet design with External and internal Channels on Torus topology. Team informed that 200ms delays have been added after kernel launches and print statements are included and Concat 3b is not working. Tobias suggested to try synthesis from a different seed and not specifying channel depth and other possible alternatives.


##### Project decisions
- Team discussed the tentative dates for Project presentation(late September ~25th) and Report submission(early october).
- Tobias suggested the team to consider a Hybrid design for GoogleNet  with internal channels and MPI calls or Global memory with External IO channels.

### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| Issue 1| - |
| Issue 2 | - |

|Action| Owner|Due Date|
|:--- | :--- | :---: |
| GoogleNet Optimization version 2 | Chiranjeevi, Alina, Adesh, Arathy, Aayush  |
| GoogleNet & ResNet Optimization Synthesis  | Aayush |
| ResNet optimization | Amey, Nikitha, Anshul |
