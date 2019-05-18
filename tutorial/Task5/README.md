# Task 5

### Getting started
- Inspect the contents of the Task 5 folder.
- Create a separate folder for your solution `Task5Solution<YourIMTName>`, where you copy all the files from the `Task5` folder. Also work in a separate branch, push your solution to the remote repository and create a merge request.
- There is a host code `main.cpp` using also `utility.cpp` and `utility.h`, very similar to the previous examples. Refer to lines 84-89 to find out which argument this host code supplies to the kernel.
- There is a `makefile` to build and run your code. Find about the different targets that you can invoke. The makefile uses targets with wildcards that allow you to execute different kernel variants with a single command. Execute `make run` and `make run-2nu` and find out how you execute different kernels here.

### Reports
- Generate the report for the kernel file `summation.cl`, open the `summation/reports/report.html` file with your webbrowser and locate the initiation interval of the loop. Which makefile command do you use?

### Shift Register Optimization
- Copy `summation.cl` to `summation-sr.cl`, where you will create an optimized version by implementing a shift register to relieve the data dependency.
	- Delete the line inside the loop that sets the result. 
	- Outside of the loop, define a double array of II+1 (II = 10) elements. Name the array `sum_copies[11]`.
	- Initialize all members of the sum_copies array to 0 using a loop.
	- Inside the existing loop, create the double variable named `cur` and assign it to the same calculation that was performed in the original kernel, using `sum_copies[10]` instead of result.
	- After the previous step, shift all values of `sum_copies` up. Create a loop that starts at the top and decrements. Set each element to be the value of the previous element in the array.
	- After the shift step, set `sum_copies[0]` to `curr`.
	- After the calculation loops, create a new loop that loops through all 11 elements of `sum_copies` and accumulate it to `result`.
	- Add the line `#pragma unroll` before every loop **except** the one that goes from 0 to vectorSize.
- Save your kernel file.
- Build and execute your kernel file with `make run-sr`.
	- If there are any errors, resolve them before continuing to the next step.
	- If the output does not show `VERIFICATION PASSED`, fix the functionality of your kernel.
	- You can refer to the optimization slides for the shift register pattern.
	- You can use `printf` statements to debug your kernel. Make sure to remove them before the next step.
- Generate the report for `summation-sr.cl`, open the `summation_sr/reports/report.html` file with your webbrowser and locate the initiation interval of the loop.
	- If the II is not 1, identify the reason for this and try to fix the shift register pattern accordingly.

### Parallelization Take 1
- Copy `summation-sr.cl` to `summation-sr_u.cl`, where you will try to parallelize the the design.
- Add a `pragma unroll 16` to the loop that has remained sequential.
- Generate the report for `summation-sr_u.cl`, open the `summation_sr_u/reports/report.html` file with your webbrowser and locate the initiation interval of the loop. What is the new II, what impact on overall performance will this have?
- The unrolling creates 16 copies of the hardware that need to accumulate into the the variable `cur`. This latency can be overcome with a shift register. We already have a shift register in place.
- Copy `summation-sr_u.cl` to `summation-bsr_u.cl`, where you will increase the shift register size to overcome the increased latency.
- Modify the size of your shift register and all associated loops until the II goes down to 1.
- The shift register has become fairly large. Maybe we can find a more efficient design?

### Parallelization Take 2
- Generate the report for the kernel file `summation-2nu.cl`, open the `summation_2nu/reports/report.html` file with your webbrowser and observe the initiation interval of the outer loop. The inner loop is fully unrolled with the same amount of parallelism as we applied in the previous step.
- Copy `summation-2nu.cl` to `summation-2nu_lv.cl`, where you will reduce the II using a local variable.
- Apply the pattern from the *Optimization OpenCL for Intel FPGAs.pdf*, slide 45 to the loop nest.
- Generate the report for `summation-2nu_lv.cl`, open the `summation_2nu_lv/reports/report.html` file with your webbrowser and locate the initiation interval of the loop. What is the new II?
- Copy `summation-2nu_lv.cl` to `summation-2nu_lv_sr.cl`, where you will add a shift register to the outer loop to overcome the remaining latency.
- Apply the shift register pattern from `summation-sr.cl` to accumulate your new local result variable inside the outer loop into the overall `result` variable. Match the shift register size to the II from `summation-2nu_lv.cl`.
- Generate the report for the kernel file `summation-2nu_lv_sr.cl`, open the `summation_2nu_lv_sr/reports/report.html` file with your webbrowser and verify that the II is now 1.
- Compare the resource usage of `summation-2nu_lv_sr.cl` with that of `summation-bsr_u.cl`. Which design is superior?

### Submission
- Summarize your findings in a `Findings.md` file
- Commit and push all design files to your git branch and create a merge request.
	- `summation-sr.cl`
	- `summation-sr_u.cl`
	- `summation-bsr_u.cl`
	- `summation-2nu_lv.cl`
	- `summation-2nu_lv_sr.cl`