# Task 5: Optimization Task

The task was to implement the shift register and local variable technique in order to optimize the resource usage by kernel code
Files


# Files:
1. `summation-sr.cl`
2. `summation-sr_u.cl`
3. `summation-bsr_u.cl`
4. `summation-2nu_lv.cl`
5. `summation-2nu_lv_sr.cl`


## Makefile Commands:
**This command enables to execute different kernel variants**

> summation_%.aocx: summation-%.cl
aoc -march=emulator summation-$*.cl
-rm summation_$*.aoco

**This command enables us to generate report for different kernel variants**

> report-%: summation_%.aoco
	@echo ""

Example : `make run-sr` | `make report-sr`



## Parallelization Take1 :

Initially in `summation-sr.cl` we had II of 10. After applying the shift register technique we got an II of 1. The next task was to `#pragma unroll 16 ` the outer loop in `summation-sr_u.cl`. We got an II of 17 which we needed to bring down to 1. Through trial and error method in `summation-bsr_u.cl` we were able to achieve II of 1 by setting `#define II 180` register size to 180 in the kernel code

## Parallelization Take2 :
In `summation-2nu.cl` we get an II of 177 which is very high. To reduce this we then use the local variable technique in `summation-2nu_lv.cl` which gave an II of 10. To further reduce this to 1 we then use the shift register technique in `summation-2nu_lv_sr.cl` to get an II of 1


## summation-bsr_u.cl   vs   summation-2nu_lv_sr.cl

There is significant difference in resource usage by `summation-bsr_u.cl` and `summation-2nu_lv_sr.cl`. 

 - In terms of ALUs and FFs `summation-bsr_u.cl` was very expensive (**ALUs** =105% ,  **FFs** =49% ) as compared to `summation-2nu_lv_sr.cl` (**ALUs** =25% ,  **FFs** =19% )
 - In terms of RAMs and DSPs both had almost equal usage standing at (**RAMs** =19% ,  **DSPs** =5% )


