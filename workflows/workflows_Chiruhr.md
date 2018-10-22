1) Which parts of the infrastructure do you expect to use within the project?
- We will be using CC Cluster infrastructure within the project. 

2) Which tools, FPGAs and boards do you expect to use within the project?
-cc-7
•	FPGA boards and FPGAs
•	Nallatech 385A board with Intel/Altera Arria 10 GX 1150 FPGA
•	Alpha Data 7v3 board with Xilinx Virtex-7 XC7VX690T FPGA

3) Which of the sanity checks from the FPGA documentation can you perform 
this machine, does it contain FPGAs?
- aoc -version, aocl version, quartus_cmd -version, aoc -list-boards checks were successful. 
aocl diagnose failed as this has to be performed in FPGA node. 

4) What is the path to your mounted user home on this system? What is 
your quota here 
/upb/departments/pc2/users/c/chiruhr is the path mounted user home. 
5 GB is my quota.

5)  Can you access and edit the .bashrc file there?
Yes, I can access this file. 

6) In the file system, go to 
/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2 and setup a 
local working directory with your IMT username. What is your quota here 
(use again `df -h` to find out)?
= 13TB

6) Do you prefer ssh or xrdp for your normal workflow? 
- Both work fine with my system. SSH works faster. 
 Where do you want to keep cloned versions of the git repository?
 - For availability reasons, I would like to keep at both Git and Local. 
 - For syncing between the repos, I am using DeltaCopy. 

7) Where do you want to keep cloned versions of the git repository?
     - One copy in the cluster infrastructure 
     - Another copy on your local machine for writing and editing 
documentation, 

8) If so, how will you synchronize between the two 
repositories?
	- sshfs and winfs

9)Can and will you mount the shared file system from your local 
system? How is the performance of this via cable in the lab, via 
eduroam, from your home?

- Which documentation will you use frequently, how will you access it, 
do you need local copies of the relevant pdfs?



