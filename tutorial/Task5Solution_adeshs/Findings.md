# Task 5 - Kernel Optimizations

### Report Generation  
Following two commands are used to generate the report and to execute the OpenCL summation kernels.  
- `make run-<%>` to compile the kernel and execute the host code
- `make report-<%>` to generate the HTML report for the kernel.

### Shift Register Optimization
- In the unoptimized summation kernel code, II is 10 and Latency is 32.
- Shift Register with 11 elements was applied to the kernel to reduce II to 1 and Latency=25
- All the loops except the one that goes till vectorSize was unrolled.

### Parallelization 1
- After adding pragma unroll 16 to the sequential loop, II was increased to 17. This will Increase the loop latency to 212.
- So to reduce II back to 1, I increased the size of the shift register to 177.
- I decided to increase the shift register size to 177 because the initial summation code with unroll factor of 16 in the inner loop had II=177.
- This approach is not Resource efficient. Since the size of shift register was increased to 177, estimated LUTs Usage was 89%.

### Parallelization 2
- summation-2u kernel had initiation interval of 177.  
- Using local variable, serial region issue is eliminated. The new II = 10
- To reduce II to 1, a shift register of 11 elements is used.
- Now the Estimated LUTs Usage is 18%, Hence this approach is better than the kernel in Parallelization 1 which needed 89% of LUTs
