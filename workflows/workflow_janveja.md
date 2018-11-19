# workflow_janveja



## Task 0: Introductory Task.

#### Step 1&2

  * Got a brief overview of the documentation of the FPGA/Custom Computing infrastructure from [PC^2^-Wiki](https://wiki.pc2.uni-paderborn.de/display/FPGAIn/Infrastructure+Overview).		 

*  Expect to work at the Custom Computing infrastructure through the CC front end.
*  Checked the Git documentation. Able to work comfortably with "Git" and "Command-Line".    

#### Step 3

* Able to connect to the CC front end using both "ssh" and "xrdp" network protocols. 
* Able to clone and access the project repository pg-custonn2-2018 in my working directory called "janveja". 
* Performed various sanity checks mentioned in the PC^2^  Wiki documentation at the CC front-end and checked my quota here. 
* Able to access and edit the .bashrc and .bash_profile files.  

#### Step 4

* Connection to CC front-end through both "ssh" and "xrdp" is working well through home (VPN set-up to access xrdp from home) and university networks.
* I currently have one clone of the repository at the CC front-end cluster infrastructure in order to use the Open CL tools available there. 
* I also have a 2nd clone of the repository on my local machine for writing/editing documentation. 
* Regular work on both clones of the repository through my laptop and CC front-end will automatically ensure that both the copies stay synchronized and up to date with the remote  repository. Since regular fetching of updates will keep taking place before committing and pushing my own changes. 
* I will be referring to the OpenCL "programming guide" and "best practices" guide regularly and therefore have a local pdf of both of them.



## Task 1: OpenCL Host-side Code & Kernel Code.

#### Exercise 1

* Able to fetch the exercise code templates and project files from remote using "git pull".
* Able to activate and launch the eclipse environment at cc front-end accessed through xrdp.
* Set the workspace in eclipse to "path to repository/tutorial/Task1". 
* A project named "SimpleOpenCL" is open in the "Project Explorer" pane. 
* In **Step 1** - examined the project settings. The "aocl compile-config" command gave the configuration needed for compiler settings.
* In the "Tools settings" tab in eclipse within "GCC C++ Linker", these libraries - "alteracl", "nalla_pcie_mmd" and "elf" have been set up for the linker.
* Similarly examined the other project settings as per the instructions given. 
* **Exercise Summary: ** Practiced writing Open CL host-side code that moved content into the device memory.
* **note:** Everything we wrote up until this point is considered setup code using OpenCL platform layer APIs. This is code that only needs to be written once and can be used in may application scenarios. In your own code, you would likely store the setup code in a function so it can be easily reused. You will need a more flexible approach to handle different platform names and different numbers of devices though.

#### Exercise 2

* Here we write an OpenCL Kernel and launch it from the host-code that you started writing.
* you need to rerun the "aoc" compiler whenever you edit the kernel file.
* **note:** If we were not running in emulation mode, the variable named "mybinaries" would store the information from the ".aocx" file used to program the FPGA. However, in emulation mode, the aocx file is just a software library that can be dynamically linked with the host code.

#### Exercise 3

* In this exercise, we will convert the kernel from exercise 2 into the an NDRange kernel. 



## Task 2: bashrc, makefile and Research part. 

* Configured the **bashrc file** with alias, environment variables and noted the difference from **bash_profile file**. 
* The purpose of this **makefile** is to allow building host code, kernel binaries and kernel reports (html reports) from command line, for both the 17.1.2 and the 18.0.1 tool chains. It is especially useful when compiling multiple files, making the process less tedious. 
* We use the make file to create executables and object files of our C++ files and also include a "clean" command at the end to remove the executable/object files after compilation and report generation. Only changed files are compiled when you execute "make all". 
* We can also declare variables in the makefile and further detailed information is available here: [GNU Make](https://www.gnu.org/software/make/manual/make.html). 
* Added a **.gitignore** file to prevent unwanted system generated files from being added again to every git commit. 
* **Channels**: The Intel FPGA SDK for OpenCL channels extension provides a mechanism for passing
  data between kernels and synchronizing kernels with high efficiency and low latency.
  Attention: If you want to leverage the capabilities of channels but have the ability to run your
  kernel program using other SDKs, implement your design using OpenCL pipes instead.
  * The Intel FPGA SDK for OpenCL channels extension allows kernels to communicate
    directly with each other through FIFO buffers.
    Implementation of channels decouples data movement between concurrently
    executing kernels from the host processor.
  * Data written to a channel remains in a channel as long as the kernel program remains
    loaded on the FPGA device. In other words, data written to a channel persists across
    multiple work-groups and NDRange invocations. However, data is not persistent across
    multiple or different invocations of kernel programs that lead to FPGA device
    reprogramming.
* **Pipes**: They provide a mechanism for passing data to kernels and synchronizing kernels with high
  efficiency and low latency.
  Implement pipes if it is important that your OpenCL kernel is compatible with other
  SDKs.
  * OpenCL pipes allow kernels to communicate directly with each other via FIFO buffers.
  * Implementation of pipes decouples kernel execution from the host processor. The
    foundation of the Intel FPGA SDK for OpenCL pipes support is the SDK's channels
    extension. However, the syntax for pipe functions differs from the channels syntax.
* CNN Lecture Notes made separately from the **Stanford University CS231** YouTube lectures.



## Task 3: Group Task - MNIST Handwritten Character Classification. 

* 



## Task4: Git best practices and Knowledge base documentation. 

* Made a file for both **Git** and **Command Line** highlighting the syntax and workings of the useful commands with examples. Also got practical experience by completing the interactive courses for both on Codecademy. 
* Solved a **git merge conflict** that happened naturally on the cc-front. It was with my workflow document. I was also able to sort the merge conflict taking place due to system generated files from "eclipse" by carefully going through the git messages and using the "git checkout --theirs filename" command. 
* set the Git global --color ui configuration to "true". 
* The given Git command sequence can avoid merge conflicts as we check the status of the branch and add all the necessary changes to the "staging area" before finally committing the changes and pushing them to the remote repository. 
* Re-basing the develop branch has been implemented by me in my Git course on Codecademy, but not in my project repository. Essentially "git pull -rebase" is an option of the git pull command that copies the remote commit sequence and appends them to the master branch. Instead, the regular merge makes a single merge commit for all the remote commits made, along with a combined log message.



## Task5: Applying Optimization Techniques. 

* 





***---------------X---------------*** 

