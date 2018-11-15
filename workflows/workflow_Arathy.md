# CustoNN2
### Task 2
##### Which parts of the infrastructure do you expect to use within the project?
> CC cluster infrastructure

##### Which tools, FPGAs and boards do you expect to use within the project?
>FPGA boards , Nallatech 385A board with Intel/Altera Arria 10 GX 1150 FPGA

### Task 3
##### Which of the sanity checks from the FPGA documentation can you perform this machine, does it contain FPGAs? 
> aoc -version
aoc -list-boards 



##### What is the path to your mounted user home on this system?
> /upb/departments/pc2/users/a/Arathy

##### Can you access and edit the .bashrc file there?
> Yes. I can access as well as edit the file. I have created shortcuts for gitpull and gitpush.

### Task 4

##### ssh or xrdp for your normal workflow
> I prefer ssh over xrdp for my normal workflow as there is no need to use VPN when connectiong from home. When it comes to xrdp, we need to use the vpn to connect to the cluster.

##### Where do you want to keep cloned versions of the git repository?
> One copy in cluster infrastructure and the other copy in local machine.

##### how will you synchronize between the two repositories ?
> By using GitPull

##### Can and will you mount the shared file system from your local system? 

##### How is the performance of this via cable in the lab, via eduroam, from your home?

##### Which documentation will you use frequently, how will you access it,do you need local copies of the relevant pdfs?
> I refer https://docs.gitlab.com/ee/user/

### Documentation and Knowledge Base
##### FPGAs and OpenCL SDK tool versions
> Channels pass data with high efficiency and low latency.
3 types of channels : 
-kernal to kernal - transfer data directly using on chip pathways without using global memory
-I/O to kernal - have an I/O pathway to transfer the data to kernal
-host to kernel - allow the data to be written directly to the kernal without having the data go to the global memory.
The data passed within the channels are held in FIFO memory.
Once the channel is declared within a kernel, we use it within the kernel code by reading and writing to the channel
Two methods of using the channel : blocking and non-blocking
Blocking
Kernel execution does not happen until the data has been transferrred into or out of the channel.
Which means the channel must have data present before the kernel moves on
For write the channel must have room in it's FIFO for data to be written.
Non-blocking manner
Code will continue to execute even after it encounters the function for reading or writing the channel whether
or not data was actually transferred.
Source : https://www.youtube.com/watch?v=_0RtAKeRl00&t=809s

##### aoc command line options
> 



##### FPGAs and OpenCL SDK tool versions
> I have used two versions of SDK : 17.1.2 and 18.0.1

#### Git best practices
##### Git status
> This command is used to check the current status of the repository.
Source = https://githowto.com/checking_status

##### git add -n
> Don’t actually add the file(s), just show if they exist and/or will be ignored.
Source : https://git-scm.com/docs/git-add

##### git add
> This command can be performed multiple times before a commit. It only adds the content of the specified file(s) at the time the add command is run.
To include subsequent changes in the next commit, git add must be run again to add the new content to the index.
Source : https://git-scm.com/docs/git-add

##### git reset
> Reset current HEAD to the specified state
Source : https://git-scm.com/docs/git-reset

##### git commit -m "Message"	
> Perform commit with the given message 
Source : https://git-scm.com/docs/git-commit

##### Git rebase
> Used to add commits from one branch to another branch
Source : https://git-scm.com/docs/git-rebase

 

