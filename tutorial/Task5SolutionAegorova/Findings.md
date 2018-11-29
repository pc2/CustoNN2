# Task 5


### Getting started
- There is a host code `main.cpp` using also `utility.cpp` and `utility.h`, very similar to the previous examples. Refer to lines 84-89 to find out which argument this host code supplies to the kernel.
	- We have 3 parameters. The first is double array for input, the second one is double array for output and the third one is variable vectorSize.

- There is a `makefile` to build and run your code. Find about the different targets that you can invoke. The makefile uses targets with wildcards that allow you to execute different kernel variants with a single command. Execute `make run` and `make run-2nu` and find out how you execute different kernels here.
	- By using % sign.

### Reports
- We use `make report-%` function to create reports for different kernels. 

### Parallelization Take 1

- When I added a `pragma unroll 16` line I got II as 17 and the latency increased too.
- After that I increased the shift register by 18 and II goes down to 1. 

### Parallelization Take 2

- Before I started I have a II as 177.
- Then I applied pattern from the slide 45 and II goes down to 10.
- Then I applied the shift register pattern and II goes down to 1.


- Based on the resource usage table the summation-2nu_lv_sr kernel spends less resources than summation-bsr_u kernel despite the fact that they're both deciding the same task. As I understood in first case we use the variable INNER to increase the II but in the second case we use pragma unroll 16. The minimum size of array for shift register in first case less than in the second to get II equal to 1.
