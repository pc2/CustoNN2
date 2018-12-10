# Task5 findings

Arguements in main.cpp
>  Lines 84-89 has 3 buffers : 1 input buffer, 1 output buffer and vectorsize(number of elements)

Execution of kernels
> make run-sr <br /> make report-sr <br /> 
> the above two commands are used for summation-sr.cl <br />
> Similarly we do the same for other kernel files

Parallelization task1
>After the addition of pragma unroll 16, initiation interval became 17 (II=17)
>By tncreasing the size of the shift register to 20, initiation interval became 1. (II=1)
>However this leads to excessive resource usage and hence not efficient.



Parallelization task2
>After the generation of summation_2nu/reports/report.html, initiation interval was found to be 177. <br /> After applying the local optimization technique, initiation interval became 10. <br />
>After adding the shift register to the outer loop, initiation interval reduces to 1.  <br />
>summation-bsr_u.cl is better than summation-2nu_lv_sr.cl as there is less resource usage compared to the latter. 

