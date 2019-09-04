|              |                  |              |           |                 |           |                     |          |          |
|--------------|------------------|--------------|-----------|-----------------|-----------|---------------------|----------|----------| 
| File name   | Total Operations | Total Cycles | Fmax (in Hz)      | Total Exec Time(measured)| Ops/cycle (measured) |Ops/cycle (estimated) |Global Memory (r/w) | Ops/byte | 
| inception0 | 1.13G        | 136M       | 220.8M     | 620 ms    | 8.25       | 8  | 1MB| 1.1K     | 



  
    
Kernel wise analysis :   

|              |                  |              |           |                 |           |                     |          |          |
|--------------|------------------|--------------|-----------|-----------------|-----------|---------------------|----------|----------| 
| Kernel   | No.of Operations | Global Mem (r) | Global Mem (w)      | Exe time(measured)| Ops/cycle (measured) |Ops/cycle (estimated) |Channel Read|Channel Write | Ops/byte | 
| Padding_Conv2d_1a_7x7_Conv2D | 0       |  588KB      | 0     |  1.04ms   |0       | 0  |0 | 614KB     | 0 |

