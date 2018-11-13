# workflow_janveja

## Task 0: Introductory Task

### Step 1&2

  * Got a brief overview of the documentation of the FPGA/Custom Computing infrastructure from [PC^2^-Wiki](https://wiki.pc2.uni-paderborn.de/display/FPGAIn/Infrastructure+Overview).		 

*  Expect to work at the Custom Computing infrastructure through the CC front end.
*  Checked the Git documentation. Able to work comfortably with "Git" and "Command-Line".    

### Step 3

* Able to connect to the CC front end using both "ssh" and "xrdp" network protocols. 
* Able to clone and access the project repository pg-custonn2-2018 in my working directory called "janveja". 
* Performed various sanity checks mentioned in the PC^2^  Wiki documentation at the CC front-end and checked my quota here. 
* Able to access and edit the .bashrc and .bash_profile files.  

### Step 4

* Connection to CC front-end through both "ssh" and "xrdp" is working well through home (VPN set-up to access xrdp from home) and university networks.
* I currently have one clone of the repository at the CC front-end cluster infrastructure in order to use the Open CL tools available there. 
* I also have a 2nd clone of the repository on my local machine for writing/editing documentation. 
* Regular work on both clones of the repository through my laptop and CC front-end will automatically ensure that both the copies stay synchronized and up to date with the remote  repository. Since regular fetching of updates will keep taking place before committing and pushing my own changes. 
* I will be referring to the OpenCL "programming guide" and "best practices" guide regularly and therefore have a local pdf of both of them.



## Task 1: OpenCL Host-side Code

### Exercise 1

* Able to fetch the exercise code templates and project files from remote using "git pull".
* Able to activate and launch the eclipse environment at cc front-end accessed through xrdp.
* Set the workspace in eclipse to "path to repository/tutorial/Task1". 
* A project named "SimpleOpenCL" is open in the "Project Explorer" pane. 
* In **Step 1** - examined the project settings. The "aocl compile-config" command gave the configuration needed for compiler settings.
* In the "Tools settings" tab in eclipse within "GCC C++ Linker", these libraries - "alteracl", "nalla_pcie_mmd" and "elf" have been set up for the linker.
* Similarly examined the other project settings as per the instructions given. 
* In **Step 2** - opened main.cpp. In "Step 2.3" and "Step 2.5" we have exactly 1 device and only 1 platform.
* Struggling with the various programming sub-tasks. Lack of information in handling of OpenCL code and commands.  In Exercise 1 - "Practiced writing OpenCL host-side code that  moved content into the device memory."

### Exercise 2

* Here we write an OpenCL Kernel and launch it from the host-code that you started writing.

### Exercise 3

* In this exercise, we will convert the kernel from exercise 2 into the an NDRange kernel.  

### Excercise 4

* 

***---------------X---------------*** 