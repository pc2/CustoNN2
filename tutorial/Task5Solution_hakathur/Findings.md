# Task 5

## Reports
- Generate the report for the kernel file summation.cl, open the summation/reports/report.html file with your webbrowser and locate the initiation interval of the loop. Which makefile command do you use?
> make report-xyz where xyz are the suffixes for different kernels.


## Shift Register Optimization
> II is 1 after incorporating shift register . Used : ALUTS: 13 % , FFs : 10 % RAMs : 9% , DSPs : 0%

## Parallelization Take 1
- Generate the report for summation-sr_u.cl, open the summation_sr_u/reports/report.html file with your webbrowser and locate the initiation interval of the loop. What is the new II, what impact on overall performance will this have?
> The new II is ~17 . Used : ALUTS: __19__ % , FFs : 10 % RAMs : 9% , DSPs : 0%.
  This takes up 6% more of ALUTs and has a II of 17 which is inferior to the design we had for the "Shift Register Optimizaztion"


- Modify the size of your shift register and all associated loops until the II goes down to 1. (summation-bsr_u.cl)
> When size of Shift Register = 11 , II is ~17 . Used : ALUTS: 19 % , FFs : 10 % RAMs : 9% , DSPs : 0%.     
  When size of Shift Register = 22 , II is 8   . Used : ALUTS: 24 % , FFs : 15 % RAMs : 9% , DSPs : 0%.    
  When size of Shift Register = 44 , II is 4   . Used : ALUTS: 33 % , FFs : 19 % RAMs : 9% , DSPs : 0%.    
  When size of Shift Register = 88 , II is 2   . Used : ALUTS: 51 % , FFs : 26 % RAMs : 9% , DSPs : 0%.    
  When size of Shift Register = 180 , II is 1   . Used : ALUTS: 90 % , FFs : 39 % RAMs : 9% , DSPs : 0%.    
  The resource utilization when we try to bring down II to 1 is more than 100% . This cannot be compiled for the given FPGA board.

## Parallelization Take 2
- Generate the report for summation-2nu_lv.cl, open the summation_2nu_lv/reports/report.html file with your webbrowser and locate the initiation interval of the loop. What is the new II?
> The new II is ~10

- Compare the resource usage of summation-2nu_lv_sr.cl with that of summation-bsr_u.cl. Which design is superior?
> The resource usage of summation-2nu_lv_sr : Used : ALUTS: 18 % , FFs : 13 % RAMs : 10% , DSPs : 0%.
  Clearly the solution with local variable and shift register is better than the design with an increased unrolling combined with shift register.
  The RAM usage increases by 1% in 2nu_lv_sr design but the gain in terms of savings made in other resources more than makes up for this tiny difference in RAM usage.
  In future , we have to consider using local variables and shift registers to make our designs efficient and faster.





