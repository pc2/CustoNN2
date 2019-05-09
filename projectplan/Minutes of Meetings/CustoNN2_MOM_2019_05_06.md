# CustoNN2 : Minutes of the meeting
<br/>

| Date of Meeting (YYYY/MM/DD)  | 2019/05/06   |  
|:--- | :---: |  
| Time  |  16:15 - 19:00 |  
| MOM Prepared by  | Adesh  |  

| Team | |
| --- | :---: |
| Present members | All team members present | 
| Absentees | - |

### Notes and Decisions 
##### Project organization updates
- Dr. Robert gave introduction to Singularity Containers and Jupyter notebooks to the team.  
- Milestone1 report submitted
- Unable to generate OpenCL kernels for GoogLeNet using TVM. 
- Team decided to write OpenCL Kernels for GoogLeNet as backup plan if TVM is unable to generate the kernels.
 
##### Research/Individual Task Updates: Task Updates
- SimpleCNN Kernels of Milestone1 synthesis has been done. Needs to be executed on FPGA using OpenVINO plugin.
- Alternate for TVM was discussed to generate OpenCL codes, But there is no such tool to convert CNN Model to OpenCL Kernels. 
- MPI implementation for communication between nodes in Noctua cluster was discussed.
##### Project decisions
- Redistrubution of work among the team and the sub teams were shuffled.
- Following teams were formed:
	- OpenVINO FPGA Plugin with scaling capability : Adesh, Alina , Rushikesh
	- TVM (Continue research to generate OpenCL Kernels) : Chiranjeevi, Arathy
	- Kernels (Write OpenCL Kernels for GoogLeNet) : Aayush, Anshul, Nikitha, Amay, Suprajith

### Issues and Actions
| Issue | Owner | Time |
|:--- | :--- | :---: |
| TVM unable to generate OpenCL Codes | Team TVM |

|Action| Owner|Due Date|
|:--- | :--- | :---: |
| Move DLDT project to CustoNN2 GitLab project | Amay | |
| Execute SimpleCNN Kernels on Stratix 10 FPGA using OpenVINO | Team OpenVINO ||

