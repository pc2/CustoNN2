# Task 3

## Organization
- This task will be started as presence exercise on Monday November 5.
- You can work in groups of up to 4 students (max 3 groups). At first ,decide for a short name for the group.
- Every group needs to work in a separate branch `dev_<groupname>` and create a separate folder `tutorial/Task3Group<Groupname>`.
- Besides the actual sources and makefiles, also keep track of your additional findings (e.g. Part III) in `.md` files in this folder.
- Avoid committing changes that are not part of this task to this branch.

## Instructions
- The task builds upon material from an Intel OpenCL tutorial and is denoted as **Laboratory Exercise 4** there.
- You will mostly follow the instructions from their instructions ![(the Intel Laboratory Instructions).](opencl_lab4.pdf).
<object data="opencl_lab4.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="opencl_lab4.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="opencl_lab4.pdf">Download PDF</a>.</p>
    </embed>
</object>

- Start by reading through the entire exercise description.
- The dataset can be found at `/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task3_MNIST_files`. Please make sure to open these file for reading and don't overwrite them.
- Instead of the the DE-series boards from the instructions, you will target the Nallatech 385A board with Arria 10 FPGA from the previous exercises (board name for aoc: `p385a_sch_ax115`). The host code will run on the x86 CPU of cc-7, so no byte reordering should be required in Part 1.
- Be sure to first perform software emulation until you get a correct design, then generate `.html` reports while applying unrolling or other optimizations. Only when both emulation is successful and the `.html` report indicates < 100% utilization of all resources, start a  full synthesis process (only one at a time per group).
- For every generated **hardware** `.aocx` file, create a separate folder under `/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs` and put a copy of
	- the `.aocx` file
	- the corresponding `.cl` sources
	- the used build command (`aoc ...`) in a file called buildme.txt
	- the `acl_quartus_report.txt` file from the synthesis folder