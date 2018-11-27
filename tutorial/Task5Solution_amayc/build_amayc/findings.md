
### Reports

- Which makefile command do you use? 
I have edit makefile and use report and report-% (where % acts as a wildcard entries) .

### Shift Register Optimization

- For `summation-sr.cl` shift register of size 11 was used and has II of 1 and latency of 25 .
- Local variable `cur` was introduced to store results locally before writing them to result variable. 
- All loops are unrolled except calculation loop which runs till vectorsize.

### Parallelization Take 1

- We parallelize design at the calculation loop which runs till vectorsize.
- For `summation-sr_u.cl` has a II of 17 and Latency of 212 .
- For `summation-bsr_u.cl` shift register size has been set to 177 [(16*11)+1] and has latency of 183.
- We can use local memory and write values back to global memory.

### Parallelization Take 2

- We initiate local variable `result2` inside outer loop which sums all the values of the calculation from inner loop which are then added to `result` variable.
- For `summation-2nu_lv.cl` we have II of 10 .
- For `summation-2nu_lv_sr.cl` we add Shift register after computing values in local variable `result2` .
- To reduce II in `summation-2nu_lv_sr.cl` shift register of size 11 is used.
- `summation-2nu_lv_sr.cl` has ALUTs and FFs has usage of 10% and 5% respectively compared that to of `summation-bsr_u.cl` which has ALUTs and FFs at 88% and 34% respectively which is quite higher.
