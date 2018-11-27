# Task 7

## Overview
- This task will be started as presence exercise on Tuesday November 27 and continue with interactive work during the tutorial hours + some parts to be completed inbetween.
- This is a group task with implementation in small subgroups and discussion + planning in the entire group.
- Every group needs to work in a separate branch `dev_<groupname>` and create a separate folder `tutorial/Task7Group_<groupname>`.
- Besides the actual sources and makefiles, also keep track of your additional findings in an `.md` files in this folder.
- Avoid committing changes that are not part of this task to this branch.
- The task builds upon material from an Intel OpenCL tutorial and is denoted as **Laboratory Exercise 5** there.
- You will mostly follow the instructions from their instructions ![(the Intel Laboratory Instructions).](opencl_lab5.pdf).
<object data="opencl_lab5.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="opencl_lab5.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="opencl_lab5.pdf">Download PDF</a>.</p>
    </embed>
</object>

- Start by reading through the entire exercise description.
- The dataset can be found at `/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files`. Please make sure to open these file for reading and don't overwrite them.
- Instead of the the DE-series boards from the instructions, you will target the Nallatech 385A board with Arria 10 FPGA from the previous exercises (board name for aoc: `p385a_sch_ax115`). The host code will run on the x86 CPU of cc-7, so no byte reordering should be required in Part 1.

## Part I
- Form three groups
	- Arathy Ajaya Kumar, Suprajith Suresh Hakathur, Rushikesh Vinay Nagle
	- Aayush Suresh Bansal, Amay Churi, Alina Egorova, Nikhita Shivaswamy
	- Anshul Suresh Bansal, Chiranjeevi Hongalli Revanna, Adesh Shambhu
- Interactive Work in Tutorial time slot on Tuesday November 27, **completion before Monday December 3 Tutorial Session.**
- Running software code + evaluation of classifier’s accuracy.

## Part II
- Design decisions
	- single monolithic OpenCL kernel or multiple smaller kernels
	- in multiple kernels: global memory or pipes to share the data
	- utilization of local memory
	- single work item kernel or NDRange kernel

- In Tutorial session on Monday December 3
	- 45min discussion of design decisions within the subgroup from Part 1. Each subgroup decides for a set of most promising design choices.
	- 45min discussion of design decisions in plenum. Note down the decisions and expectations, then decide to distribute three different variants to the subgroups.
	- 45min start of implementation within subgroup.

- In Tutorial session on Tuesday December 4 and **to be completed before Tutorial session on Monday December 10.**
	- Continue implementation within subgroup.
	- Be sure to first perform software emulation until you get a correct design, then generate `.html` reports while applying unrolling or other optimizations.
	- Read about hardware profiling in the official documentation.
	- When both emulation is successful and the `.html` report indicates < 100% utilization of all resources, start a full synthesis process with profiling enabled.
	- Perform measurements with generated hardware design.
	
- For every generated **hardware** `.aocx` file, create a separate folder under `/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs` and put a copy of
	- the `.aocx` file
	- the corresponding `.cl` sources
	- the used build command (`aoc ...`) in a file called buildme.txt
	- the `acl_quartus_report.txt` file from the synthesis folder

## Part IIb
- Preparation: watch lectures 7-9 of Stanford c231 **before Tutorial session Monday December 10**
- In Tutorial session Monday December 10
	- Within each subgroup: open and analyze profiling report (`aoc report`).
	- Compare execution times displayed in the report with execution times measured from the host.
	- Come up with a performance model that explains the observed results. Include at least the following metrics:
		- Total number of operations performed
		- Number of operations performed per cycle
		- Initiation intervals
		- Total amount of global memory read and written
		- Global memory required per cycle
		- Global memory bandwidth as measured in report
	- Also take note of where local memory (RAM) resources are used, which fraction is required by the size of data stored, how much does local memory replication for different reasons contribute?
	
- In Tutorial session Tuesday December 11
	- Present the findings of each subgroup to the entire group.
	- Discuss bottlenecks and ideas for further optimizations for each subgroup. Each subgroup should optimize further up to a state that you consider the best obtainable design for the design decisions taken in Part II. If changes to the usage of local memory seem advisable, incorporate them. **To be completed and ready to present for tutorial session Monday January 14.**
	- Depending on the workload of the individual subgroups, have every group work on Part III, or form a task force to work in this direction.

## Part III
- **To be completed and ready to present for tutorial session Monday January 14.**, use tutorial sessions on December 17 and 18.
- Use Tensorflow to train alternative CNN topologies.
- Can you use the GPUs in Oculus for training, or do you rely on local resources.