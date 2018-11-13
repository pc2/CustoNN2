# 1. Practical part

## Customize your .bashrc file

- This task  is a good opportunity to setup shortcut functions in your .bashrc file, that will load the respective tool settings and provide a shortcut to your repository.
- Within .bashrc, you can
	- export environment variables `export GITLAB=${PC2SCRATCH}/pc2-mitarbeiter/kenter/gitlab/` 
	- use an alias for one line commands (unrelated example: `alias sxil174='source /opt/Xilinx/SDx/2017.4/settings64.sh'`)
	- declare multi-line functions 
	
```
	llicenses() {
		export LM_LICENSE_FILE="27000@kiso.uni-paderborn.de"
		export MGLS_LICENSE_FILE="27000@kiso.uni-paderborn.de"
	}
```

## Build system with makefile

- Create a new git branch from the current master branch. If you already have worked in a branch for one of the previous exercises, go back to the master branch first and pull the remote changes.
- Inside the folder Task1Exercise2Solution, create a new folder with the name build\_\<your\_imt\_username\>
- Inside the new build folder, create a file called `makefile`. The purpose of this makefile is to allow building host code, kernel binaries and kernel reports (html reports) from command line, for both the 17.1.2 and the 18.0.1 tool chains. An introduction to makefiles can be found for example [here](https://www.gnu.org/software/make/manual/html_node/Introduction.html).
	- For testing purposes, you should have open two separate terminals, one where you load the settings for 17.1.2 tools and one where you load the settings of 18.0.1 tools (refer to the wiki documentation). Switching between two tool versions inside the same terminal does not work flawlessly.
	- Test all build commands first manually in your terminal before integrating them to the makefile.

- You can start populating your new makefile with the help of `aocl makefile`. Check if the outputs of the two tool versions differ and how tool specific settings will be applied.
- Modify the initial makefile to allow building the host code for both tool versions and cleaning the built object files and binaries. You will need to use relative paths and should have all generated files within your build folder - do you need to copy/move or softlink any files for that goal?
- Add further targets to the makefile that will
	- build the emulator binary (`.aocx`) from the kernel sources (`.cl`)
	- generate the `.html` reports from the kernel sources (`.cl`)
	- execute the compiled host code with the generated `.aocx` file

## Use of .gitignore and feature branches

- You will now have a build directory with a makefile that you want to commit to the git repository and many generated files that should not be under version control. With `git add -n .` from within your build folder, you can find out which files would be added if you don't prevent that.
- Read about `.gitignore` files in your favorite git reference and create a local `.gitignore` file inside your build folder. Populate it with sufficient patterns that no generated file or folder will be added to git.
- Commit the `makefile` and the `.gitignore` to your git branch.
- Push your git branch to the remote repository.
- In the gitlab web surface, create a merge request to merge your branch back into master.

# 2. Research part

## Find out about channels and pipes

- Read about channels and pipes in the different parts of the Intel FPGA SDK for OpenCL documentation.
	- Which documents and versions are you referring to?
- What is are the differences between channels and pipes? Why do both of them exist?
- Which of them can be used in blocking and/or unblocking mode, what is their default behavior?
- Bonus challenge: can you modify the example from Task 1 (either after Exercise 2 or 3) so that the SimpleKernel only reads from one single global input buffer, receiving the values of the second global input buffer instead via a channel from another separate kernel?
	- You will need to modify both the kernel and the host code for this.

## Introduction to CNNs

- To get up to speed on CNNs, their inner workings, architectures, training and current performance, we will build on material from the Stanford University cource CS231n Convolutional Neural Networks for Visual Recognition
- For this week, watch the first two lectures on youtube
	- https://www.youtube.com/watch?v=vT1JzLTH4G4&index=1&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk
	- https://www.youtube.com/watch?v=OoUX-nOEjG0&index=2&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk
- Take some notes on the overall course contents. What was particularly new or interesting to you?