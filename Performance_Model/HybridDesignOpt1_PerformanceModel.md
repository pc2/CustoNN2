### File wise analysis :  
|              |                  |              |           |                 |           |                     |          |          |
|--------------|------------------|--------------|-----------|-----------------|-----------|---------------------|----------|----------| 
| File name   | Total Operations | Total Cycles | Fmax (in Hz)      | Total Exec Time(measured)| Ops/cycle (measured) |Ops/cycle (estimated) |Global Memory (r/w) | Ops/byte | 
| inception0 | 1.13G        | 136M       | 220.8M     | 620.4 ms    | 8.25       | 8  | x| 1.1K     | 



  
    
### Kernel wise analysis :   

|              |                  |              |           |                 |           |                     |          |          |          |          |
|--------------|------------------|--------------|-----------|-----------------|-----------|---------------------|----------|----------|----------|----------| 
| Kernel (inception0)                      | No.of Operations| Fraction of total Ops | Global Mem (r) | Global Mem (w)      | Exe time(measured)| Ops/cycle (measured) |Ops/cycle (estimated) |Channel Read|Channel Write | Ops/byte | 
| Padding_Conv2d_1a_7x7_Conv2D | 0               |0                      |  588KB      | 0|  1.04ms   |0       | 0  |0 | 614KB     | 0 |
| Conv2d_1a_7x7_Conv2D         | 256M |0.23      |  65KB                 | 0|  79.6ms   |14       | 14  |614KB | 3.21MB     | 3.93K |
| MaxPool_2a_3x3_MaxPool         | 0 |0      |  0                 | 0|  79.45ms   |0       | 0  |3.21MB | 802KB     | 0 |
| Conv2d_2b_1x1_Conv2D         | 27M |0.02      |  16KB                 | 0| 96.6ms   | 1.26       | 8| 802KB |  802KB    | 1.68K |
| Padding_Conv2d_2c_3x3_Conv2D         | 0 |0      |  0                 | 0| 97.47ms   | 0       | 0| 802KB |  841KB    | 0 |
| Conv2d_2c_3x3_Conv2D         | 848M |0.75     |  433KB                 | 0| 620ms   | 5.81       | 6| 841KB |  2.4MB    | 1.96K |
| MaxPool_3a_3x3_MaxPool         | 0 |0     | 32bits                 | 0| 620.4ms   | 0      | 0| 2.4MB |  602KB*    | 0 |

  
|              |                  |              |           |                 |           |                     |          |          |          |          |
|--------------|------------------|--------------|-----------|-----------------|-----------|---------------------|----------|----------|----------|----------| 
| Kernel (inception1)                      | No.of Operations| Fraction of total Ops | Global Mem (r) | Global Mem (w)      | Exe time(measured)| Ops/cycle (measured) |Ops/cycle (estimated) |Channel Read|Channel Write | Ops/byte | 
| feeder_3b | 0               |0                      |  0      | 0| 628.18ms   |0       | 0  |602KB* |602KB     | 0 |
| Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D | x               |x                     |  x      | x| x   |x       | x  |x |x     | x|


  
*Writes to/Reads from  external IO channels  
#### Execution time :  
All the kernels except Concat  are launched concurrently. Hence , the execution time indicates their end time rather than the amount of time the kernels were performing meaningful operations.
TODO
- [ ] We can get a rough estimate of time spent by the kernels performing their tasks from the report
- [ ] Rest of this analysis
  
#### Estimated Ops/cycle :  
This metric is estimated  as :  
sum(operations per kernel * fractions of  total operations in file) / total no of operations



