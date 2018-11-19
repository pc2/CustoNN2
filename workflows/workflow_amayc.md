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

## Resources

- Git Reference 
	- https://git-scm.com/docs

- OpenCl Channels
	- https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/hb/opencl-sdk/aocl_programming_guide.pdf
	- https://wiki.pc2.uni-paderborn.de/display/FPGAIn/Documentation+-+Intel-17.1
	- https://www.youtube.com/watch?v=_0RtAKeRl00

- AOC
	- https://www.intel.com/content/dam/altera-www/global/ja_JP/pdfs/literature/hb/opencl-sdk/aocl_programming_guide.pdf

## Task 4

### AOC commands

- `aoc -version` Gives version of the compiler.
- `aoc -list-boards` Shows list of boards available .
- `aoc --report` Generates html reports as well estimates hardware resource usage during compilation.
- `aoc -v` Direct the aoc to report on the progress of a full compilation of the kernel.
- `aoc -c <your_kernel_filename>.cl --report` To compile and generate report for the kernel.

### Git Info

- `git status` Shows us list of files to be committed, untraced files and status of commit with respect to remote branch.
- `git add -n` It's a dry run for showing which all files will be committed .
- `git add` Adds files to the commit to be made.
- `git commit -m "message"` Makes a commit along with the message of the commit.
- `git rm -r --cached some-directory` Removes unwanted files folder from remote branch.
- `git reset --hard` Reset local repo Head to last remote pull.
- `git rebase` Will apply commit on top of Master from branch.
- `git merge` Will merge branch into master repository.

### OpenCl Channels
- The Intel FPGA SDK for OpenCL channels extension provides a mechanism for passing
data between kernels and synchronizing kernels with high efficiency and low latency.
- I/O to kernel communication done without the host.
- Kernel to kernel communication done directly on-chip.
- Host to kernel communication done without global Memory.

- Features:
	- Provides fifo like communication.

- What is are the differences between channels and pipes? Why do both of them exist?
_Solution_ - AOC implements pipes as a wrapper around channels. Pipes are compatible with other sdks.
