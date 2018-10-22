## Connection Preference
- Currently prefer SSH for connecting to CC frontend.
- SSH is faster compared to XRDP
- XRDP also working with remmina-next client for Linux Ubuntu
- I have mounted the shared file system on my laptop using sshfs, giving good performance.
- I tried rysnc for mounting the files but found sshfs better.

## Repositories
- Two copies, one in my working directory on the CC cluster frontend. Another one locally on my laptop.
- I have added a remote to both the clones to pull data from the master and keep them synchronized.

## Local copies of documentation
- Currently I require git documentation.

## Task 2
- We have currently logged in to the Custom Computing(CC) Cluster. This contains Nallatech 385A board with Intel/Altera Arria 10 GX 1150 FPGA.
- Later on I expect we would be using the Noctua cluster which contains Nallatech 520N boards with Stratix 10 FPGAs.

## Task 3
- Both ssh and xrdp connections are working properly for the CC cluster. 
- Performed the sanity checks related to the Altera Compiler for Tools version 17.1.2
- I wasn't able to do sanity check on Altera Runtime environment.
- Path to my mounted user name is **/upb/departments/pc2/users/a/amayc**
- Available quota is 5Gb.
- .bashrc file can be accessed and edited.

