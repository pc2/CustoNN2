Task #1

2.
- Which parts of the infrastructure do you expect to use within the project?
Custom Computing (CC) cluster

- Which tools, FPGAs and boards do you expect to use within the project?
Nallatech 520N with Stratix 10 installation in Noctua

3.
- Which of the sanity checks from the FPGA documentation can you perform this machine, does it contain FPGAs? 
aoc -version 

- What is the path to your mounted user home on this system? What is your quota here (use `df -h` to find out)? 
/upb/departments/pc2/users/a/aegorova

- Can you access and edit the .bashrc file there?
yes. I added some aliases there.

- In the file system, go to /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2 and setup a local working directory with your IMT username. What is your quota here (use again `df -h` to find out)?
94G ?

4.
- Do you prefer ssh or xrdp for your normal workflow?
	xrdp

- Where do you want to keep cloned versions of the git repository?
	I prefer to work with cluster infrastructure.

    - Can and will you mount the shared file system from your local system? How is the performance of this via cable in the lab, via eduroam, from your home?
    I use VPN and Microsoft Remote Desktop 

- Which documentation will you use frequently, how will you access it, do you need local copies of the relevant pdfs?



Task #2



a) Find out about channels and pipes:

1. Channels and pipes communicate directly via FIFO buffers.
2. Channels work only with Intel SDK, pipes work with others.
3. The syntax for pipe functions differs from the channels syntax.
4. They both decouple kernel execution from the host processor.
5!. Unlike channels, pipes have a default nonblocking behaviour.
6. Data written to a pipe(channel) remains in a pipe(channel) as long as the kernel program remains loaded
on the FPGA device.

link: https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/hb/opencl-sdk/aocl_programming_guide.pdf

b) aoc command line options

I use aoc -help to see all commands for aoc. 

Task #4

git status - Show all my modified files and my current branch. Hint me that I can ADD files to commit phase or RM them from commit phase.
git add - After that command all files would be prepare to be in new commit.
git reset - To reset all uncommited files
git commit -m "Message"	- To create commit "Message"
git stash - To stashed all files 
git stash pop - To pull up the last stashed files
git merge name_of_branch - To create merge with another branch
git rebase - To integrate changes from one branch into another

link: https://git-scm.com/book/en/v1/Git-Basics-Recording-Changes-to-the-Repository




