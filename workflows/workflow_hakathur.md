## SSH vs XRDP
- I am able to edit files  and browse the file systems quite seamlessly using Win 10 Remote Desktop Connection . Both in lab and from home (over VPN). Even SSH is working superbly.

## Cloned Versions of the repo
- I will have one on Cluster infra and another on my laptop.


## Mounting of shared file system 
- Used the uni`s https://hilfe.uni-paderborn.de/Netzlaufwerk_einbinden_(Windows_7) to mount my shared folder on my Win 10 PC.

## Documents to be used frequently 
- The link https://www.intel.com/content/www/us/en/programmable/documentation/lit-index.html is helpful . 


## Task1 Subtask 2
- I expect to work on the **Custom Computing** part of the infrastructure.
- I think we will be using  **Nallatech 520N board with Intel Stratix 10 GX 2800 FPGA - installed in Noctua**


## Task1 Subtask3
- The sanity check mentioned by Dr. Kenter `aoc -version` can be used to see if we have all the required support software to compile our OpenCl scripts.
- We can also make use of `aoc -list-boards` . This will list the boards.

- The path to my mounted user home is _/upb/departments/pc2/users/h/hakathur
- The quota being shown to me here is 5GB
- I was able to edit .bashrc file from the above mentioned path
- The quota in _/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2_ is 15TB
 


# Task 4 Documentation and Knowledge Base

## OpenCL Error Codes
A useful place to get info about the error codes while coding in OpenCL : https://streamhpc.com/blog/2013-04-28/opencl-error-codes/

## Gitlab merge request
- I use the browser to create merge requests. So far only one merge request was issued by me as a part of the task.
 
## aoc command line options
- when -march=emulator is dropped , the aocx is created for the FPGA rather than the board. This is a very time consuming step
- when -c -report is used , we can create html report.
- const-cache-bytes=<N> where N is the cache size allows us to crete cache while compiling . This can be seen in the Task Sheet https://git.uni-paderborn.de/cs-hit/pg-custonn2-2018/blob/master/tutorial/Task3/opencl_lab4.pdf
 
## Mounting university file systems
- I made use of  https://hilfe.uni-paderborn.de/Netzlaufwerk_einbinden_(Windows_7)

## FPGAs and OpenCL SDK tool versions
- We have encountered two versions of SDKs - 17.1.2 and 18.0.1 . The boards supported are different and thus , we have to make sure that the *Makefile* can handle these changes.
https://wiki.pc2.uni-paderborn.de/pages/viewpage.action?pageId=19562930


# Task 4 Git best practices

## git status
- git status allows us to see which files are staged or modified  , untracked etc. As this page explains https://git-scm.com/docs/git-status , it helps us keep an eye on what can be pushed , commited and tracked.

> git status . 

Allows us to see the git status in the current directory.

## git add -n
- This is similar to git add --dry-run . This does not actually stage the files but gives the user a chance to see what will get added.
https://git-scm.com/docs/git-add


## git add
- This is used to stage a file so that it can be tracked
 

>   __git status__    _tells us about the files modified , ready to commit , and untracked files_      
    __git add `filename` -n__    _this does a "dry run" .It tells us if the add can happen without any problems_        
    __git add `filename`__       _stages a local file to staging area so that it can be committed later_    
    __git status__    _informs us of the file which was added in the previously steps_   
    __git reset `filename`__     _basically the opposite of git add. Unstages the added files_      
    __git status__    _Tells us the status again_      
    __git commit `filename` -m "Message"__	 _Commits the file with a message Message_    



## Channels and Pipes 
- [The Intel FPGA SDK for OpenCL channels extension provides a mechanism for passing data between kernels and synchronizing kernels with high efficiency and low latency](https://www.intel.com/content/www/us/en/programmable/documentation/mwh1391807965224/ewa1411503895045.html#ewa1411747396740)
- Channels are used to exchange data between kernels. This communication between kernels happens through FIFO buffers.
- Channels are better understood if we work in terms of Producer and Consumer. Producer "produces" data which is written to  channel and consumer makes use of the data i.e. consumer reads the data from the channel.
- If consumer does not read faster than the rate the producer writes to the channel  _stalls_ occur .We can use Intel FPGA Dynamic Profiler for OpenCL to check for channel stalls.
- If consumer reads faster than the rate the producer writes to the channel , consumer stalls. Because consumer has nothing to work on.
- A kernel can read from the same channel multiple times. However, multiple kernels cannot read from the same channel. Similarly, a kernel can write to the same channel multiple times but multiple kernels cannot write to the same channel.
- In Intel FPGA SDK ,  channels in kernel are enabled using :
  > #pragma OPENCL EXTENSION cl_intel_channels : enable
- Blocking writes to a channel is done using the signature _kernel_write_channel_ 
- Non-blocking writes are done using  _write_channel_nb_intel_ . This returns a boolean value using which the programmer can write a logic to write to a different channel (or use some other logic to keep the kernels from stalling)
- Similarly , we can have a blocking read and a non-blocking read . In blocking read , the consumer kernel stalls if there is nothing in the channel.Using nin blocking read function , the programmer gets a flag which tells if the data read from the channel is valid or not ( valid in case the channel had some data ; invalid in case the channel is empty)
- The blocking read function`s signature is : _read_channel_intel(channel <type> channel_id)_
- The non blocking read function`s signature is : _read_channel_nb_intel(channel <type> channel_id, bool * valid)_
- Pipes are similar to Channels . The main difference I could see from the documentation is that Channels are wholly supported by Interl FPGA SDKs. When working with other SDKs, it is better to work with Pipes.
- Intel FPGA SDK implementation of OpenCL pipes is partially conformant to the OpenCL Specification version 2.0
- By default, pipes exhibit nonblocking behavior. If you want the pipes in your kernel to exhibit blocking behavior, specify the blocking attribute (__attribute__((blocking))) when you declare the read and write pipes.













