# Task 5 Findings
The task was to optimize given kernel codes using loop unrolling, shift registers and local variables.

 - **Arguments to the kernel:** `main.cpp` provides 3 arguments, an input buffer, an output buffer and a variable for the no of elements. The input buffer holds a double array. The output buffer is for a single double output value.
 
 - **Makefile commands:** Simple make run and make report commands compile the `summation.cl` file and generate report for it respectively. For other kernel files, we can use for example `make run-sr` and `make report-sr` for `summation-sr.cl` and similar pattern for other files.

 - **Parallelization Take 1:** When the outer loop is unrolled by 16, the II becomes 17. This increases memory utilization as 16 copies of the variables will be needed, which if accessed from the global memory will increase the load on the memory interface. Also the II is a bottleneck due to data dependency of the shift register as its size was not enough. Upon increasing the shift register size to 180, II drops to 1. This is not efficient due to the excessive resource use. It uses 94% of the ALUTs.
 
 - **Parallelization Take 2:** II is 177 for `Summation-2nu.cl` After applying the local variable optimization to the inner loop, II drops to 10. It further drops to 1 after adding a shift register to the outer loop. This is much more efficient than the previous large shift register kernel and comparatively uses only 18% of the ALUTs while providing the same II = 1 for the task. 
