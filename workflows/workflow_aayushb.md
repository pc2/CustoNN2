
## Do you prefer ssh or xrdp for your normal workflow ?
- I prefer xrpd for connecting to the CC frontend as it provides a GUI to work with.
But xrdp is slow in opeartion as its less responsive. We need a VPN for using an xrdp connection when operating from home. Whereas in SSH we do not need a VPN and also its faster tham xrdp but does not provide a GUI

## Where do you want to keep cloned versions of the git repository?
- Two copies, one in my working directory on the CC cluster frontend /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/aayushb. 
Another one locally on my laptop.
- I have added a remote to both the clones to pull data from the master and keep them synchronized.

## Mount shared file system from the local system 
**TODO**

##  Which documentation will you use frequently, how will you access it, do you need local copies of the relevant pdfs?
**TODO**




## Task 2
- We have currently logged in to the Custom Computing(CC) Cluster. This contains Nallatech 385A boards with Intel/Altera Arria 10 GX 1150 FPGA.
- After Development phase, we will be using HPC production Noctua Cluster consisting of Nallatech 520N boards with Stratix 10 FPGAs.




## TASK 3

##  Which of the sanity checks from the FPGA documentation can you perform this machine, does it contain FPGAs?
- `aoc -version` will display the compiler version.
- `quartus_cmd -version` will give us the version of the Quartus Prime.
- `aoc -list-boards` will list down the available boards connected to the machine.
- `aocl diagnose`  could not be executed since this check has to be done on FPGA Node.

### What is the path to your mounted user home on this system? What is  your quota here 
- /upb/departments/pc2/users/a/aayushb
- Quota : 5GB

### Can you access and edit the .bashrc file there?
- Yes, .bashrc file can be accessed and edited. We add environment variables inside .bashrc file. 
- Quota in Cluster is 15TB
