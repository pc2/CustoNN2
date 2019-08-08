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

# TASK 4


# Knowledge Base 

  - OpenCL: 
1. Programming Guide: https://wiki.pc2.uni-paderborn.de/display/FPGAIn/Documentation+-+Intel-18.0
 2. 18.0 Documentation:  https://wiki.pc2.uni-paderborn.de/display/FPGAIn/Documentation+-+Intel-18.0?preview=/19563863/19563865/aocl_programming_guide-18.0.pdf
 
 

Git :

 1. Getting started with git- https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
 2. Best practices or Git: https://raygun.com/blog/git-workflow/

AOC Command line:

 - aoc --help gives documention regarding aoc command line options
 - aoc -version gives version name
 - aoc -list-boards lists out all the available boards.
 - aoc board= compiles for the specified boards

.
## OpenCL Channels 

- Channel is implemented using the FIFO mechanism for buffers. 
- These buffers are used for communication between two kernels or I/O in kernels directly
- OpenCL channels can be of blocking or non-blocking type.
- A blocking channel blocks the kernel execution for a write operation when the buffer is not free and for a read operation when the buffer is empty. 
- Whereas in non-blocking type the kernel does not wait conform to this type of operation. 

## Git best practices

 - Given Sequence of Commands:-
1. git status: Gives information about untracked files, files staged for commit, the ones that are not staged for commit and so on.
2. git add -n: Used to add specified files to git so that they will be tracked and staged for commit.
3. git reset: Reset a specific commit.
4. git commit -m "message": Commit the staged files with a message describing the commit.



It is a good practice to run git status after adding  file(s) to check if they have been correctly staged. git diff gives us information about conflicts while merging branches or pulling from the remote at the exact lines in the files in question.


- git rebase: It is used for integrating two branches, similar to merge.
- git log: It provides the commit history of the repository in a sequential time ordered fashion. Each commit, its author, their commit message and time is displayed.


## Genrating profile.mon file 

Command : 
>  aocl report /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/googlenet_bitstreams/inception0.aocx  profile_inception0.mon /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/googlenet_bitstreams/inception0.source


