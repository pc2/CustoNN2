## Report Generation
- For report generation two commands can be used . Following are the commands.
    - Ex1. make summation_%.aoco
    - Ex2. make report-%.

## Shift Register Optimisation
- Shift register optimisation with array of size[II+1] is implemented.
- Before implementing the Shift Register , the II of summation.cl file was 10 with Latency=32. After implementing the Shift Register, the II of summation_sr.cl becomes 1 with Latency=25. 

## Parallelisation Take 1
- After adding pragma unroll 16 to the outer loop in the summation_sr_u.cl the II shoots up to 17 with Latency=212. This is due to the fact that after doing that 16 copies of the variable is needed which then increases resource consumption.
- Since the II is 17 we need to increase the Shift register size to more than 170(approx 180). Hence after doing that the II of summation_bsr_u.cl goes down to 1 with Latency=183 , but this will not be that efficient due to high resource use.

## Parallelisation Take 2
- The II of summation-2nu.cl is 177 with Latency=522 since the inner loop is fully unrolled along with  dependency on the variable result.
- To apply the local Optimisation Technique we have used a variable result1 which computes local sum.After doing that the II of summation-2nu_lv.cl goes down to 10 with Latency=355.
- To overcome the latency we apply the Shift Register Pattern the same way we did before with the size of 10. After doing this the II of summation-2nu_lv_sr.cl goes down to 1 with the Latency= 336.
- The design of summation-2nu_lv_sr.cl is superior as compared to summation-bsr_u due to the fact that the latter one has the ALUTs usage of 90% whereas the one former one has the ALUTs usage of 18%.

