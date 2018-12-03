# Task 5

### Reports
Q1. Execute make run and make run-2nu and find out how you execute different kernels here.
- make run executes summation.cl kernel. 
- make run -2nu runs the summation-2nu.cl kernel

### Shift Register Optimization

Q2. Generate the report for the kernel file summation.cl, open the summation/reports/report.html file with your webbrowser and locate the initiation interval of the loop. Which makefile command do you use?
- make run is the command used to generate the report.
- Initiation Interval of the Loop is 10.

Q3. If the II is not 1, identify the reason for this and try to fix the shift register pattern accordingly.
- The report says the II for the shift register kernel is 2 with the following reason.
- The critical path that prevented successful II = 1 scheduling:
- 10 clock cycles Double-precision Floating-point Add Operation (summation-sr.cl: 20)
	cur = (input[i]*0.5) + sum_copies[10];

### Parallelization Take 1
Q4.Generate the report for summation-sr_u.cl, open the summation_sr_u/reports/report.html file with your webbrowser and locate the initiation interval of the loop. What is the new II, what impact on overall performance will this have?
- The II interestingly got increased to 19 and Latency has increased too. 

Q5.The shift register has become fairly large. Maybe we can find a more efficient design?
- I used Shift register of Size 11 in previous kernels. This time Defined the size using a variable M.
- I set the M to the powers of 2. for 64, II was 3. for 128 it was 2. But the Resource initialization was exponentially increasing. 
- Finally I got the II for size 256. Later I reduced it to 190 and It was still 1. 

The shift register has become fairly large. YES we should find a more efficient design.

### Parallelization Take 2

Initially, summation-2nu.cl has II of 177. 

Q6. open the summation_2nu_lv/reports/report.html file with your webbrowser and locate the initiation interval of the loop. What is the new II?
- The II has reduced from 177 to 10. 

Q7. open the summation_2nu_lv_sr/reports/report.html file with your webbrowser and verify that the II is now 1.
- The II has reduced to 1 just for Shift Register Size of 16

The design - Using Local Variable is the best design. 




