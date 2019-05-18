
## Connection types and performance
- Currently prefer ssh for connecting to CC frontend.
- Xrdp also working with remmina-next client for Arch Linux
- I have mounted the shared file system on my laptop using sshfs, giving good performance.

## Repositories
- Two copies, one in my working directory on the CC cluster frontend. Another one locally on my laptop.
- I have added a remote to both the clones to pull data from the master and keep them synchronized.

## Local copies of documentation
- Currently I require git documentation, I have printed the links provided in the e-mail to pdfs stored locally.

## Task 2
- We have currently logged in to the Custom Computing(CC) Cluster. This contains Nallatech 385A boards with Intel/Altera Arria 10 GX 1150 FPGA.  
- Later on I expect we would be using the Noctua cluster which contains Nallatech 520N boards with Stratix 10 FPGAs.

## Task 3
- Both ssh and xrdp connections are working properly for the CC cluster. 
- Performed the sanity checks related to the Altera Compiler as given in the documentation.
- Path to my mounted user name is **/upb/departments/pc2/users/r/rnagle**
- .bashrc file can be accessed and edited.

## Channels and Pipes
- Channels and pipes are both FIFO buffers used for communication between two kernels or I/O in kernels directly, independent of global memory and host processor.
- Pipes are used when there is a requirement to facilitate communication between kernels with different SDKs.
- Pipes are non-blocking by default. Channels can be both blocking and non-blocking.
- A blocking channel blocks the kernel execution for a write operation when the buffer is not free and for a read operation when the buffer is empty. The kernel thus has to wait until these conditions are satisfied. 
- A non-blocking channel does not wait for the buffer to be free for a write operation and immediately attempts to write the value, returning whether the operation was successful or not. Similarly for read operation, it does not wait for the buffer to contain data. 
- I referred to the aocl programming guide from PC2 wiki for version 18.0 to obtain the above information. 

## Convolutional Neural Networks course
- Computer Vision has a number of challenges such as object recognition, feature extraction, scene analysis, image classification etc. These tasks are easily carried out by a human brain whereas a computer needs complex models as an image is represented as a matrix of pixels.
- Image-net, a large dataset containing milions of images and thousands of categories is used for testing classification algorithms. CNN models have been performing well and both accuracy and number of layers are increasing. 
- KNN - Simplest classifier, cannot be used in real applications as testing is slow and distance measures do not work well on images.

## Knowledge Base

 - **OpenCL:-**
	 - Programming Guide: https://wiki.pc2.uni-paderborn.de/display/FPGAIn/Documentation+-+Intel-18.0
	 - Concepts: https://wiki.pc2.uni-paderborn.de/display/FPGAIn/Documentation+-+Intel-18.0?preview=/19563863/19563865/aocl_programming_guide-18.0.pdf
 - **Git:-**
	 - Basics: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
	 - Best Practices: https://raygun.com/blog/git-workflow/
 
 - **Connecting to CC Front-end:-**
	 - SSH: `ssh -Y username@fe-1.cc.pc2.uni-paderborn.de`
	 - SSHFS: `sudo sshfs -o allow_other,defer_permissions username@fe-1.cc.pc2.uni-paderborn.de /mnt`

## Git Best Practices

 - **Colored output in Git:** In order to enable colored output we can edit the `~/.gitconfig` file or run the command `git config --global color.ui always`
 - **Given Sequence of Commands:-**
	 - `git status`: Gives information about untracked files, files staged for commit, the ones that are not staged for commit and so on. 
	 - `git add -n`: Used to add specified files to git so that they will be tracked and staged for commit. 
	 - `git reset`: Reset a specific commit.
	 - `git commit -m "message"`: Commit the staged files with a message describing the commit.
	 
 - It is a good practice to run `git status` after adding  file(s) to check if they have been correctly staged. `git diff` gives us information about conflicts while merging branches or pulling from the remote at the exact lines in the files in question. 
 - `git rebase`: It is used for integrating two branches, similar to merge. Rebase however provides cleaner merge history that is easier to understand. 
 - `git log`: It provides the commit history of the repository in a sequential time ordered fashion. Each commit, its author, their commit message and time is displayed.

	  

 


