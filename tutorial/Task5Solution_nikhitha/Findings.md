## REPORT GENERATION
- Two commands given in the makefile , "make summation_%.aoco" & "make report-%". % describes parameter of the file.

## SHIFT REGISTER OPTIMIATION
- Shift register of size II+1 is used.
- Before optimization II=10 , Latency=32.
- After optimization II=1 , Latency=25.

## PARALLELISATION TAKE 1 
- Pragma unroll 16 makes II=~17 , Latency=212.
- To reduce II we increase Shift register size=~180 , then II=1 , Latency=183 but increased resource consumption.

## PARALLELISATION TAKE 2
- summation-2nu.cl has II=177 , Latency=522.
- Local optimisation technique reduces II=10 , Latency=355.
- Applying Shift register along with Local optimisation further reduces II=1 , Latency=336.
- summation-2nu_lv_sr has ALUT's usage=18% , summation-bsr_u has ALUT's usage=90% . Hence first design is better.
