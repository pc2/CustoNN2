From report.html in emulation mode

| Layer Name   | # Ops Calculated | # Ops Litreature | Difference  | Logic Utilization | ALUTs | Dedicated Logic Registers | Memory blocks | DSP Blocks |
|--------------|------------------|------------------|-------------|-------------------|-------|---------------------------|---------------|------------|
| Inception_3b | 123433408        | 128000000        | 4566592     | 74                | 39    | 37                        | 42            | 23         |
| Inception_3c | 268660864        | 304000000        | 35339136    | 65                | 34    | 33                        | 37            | 23         |
| Inception_4b | 72269568         | 73000000         | 730432      | 65                | 34    | 33                        | 36            | 23         |
| Inception_4c | 84230144         | 88000000         | 3769856     | 64                | 33    | 32                        | 36            | 23         |
| Inception_4d | 96176128         | 100000000        | 3823872     | 64                | 33    | 32                        | 36            | 23         |
| Inception_4e | 113245248        | 119000000        | 5754752     | 64                | 34    | 32                        | 36            | 23         |
| Inception_4f | 158398336        | 170000000        | 11601664    | 64                | 34    | 32                        | 36            | 23         |
| Inception_5b | 48331904         | 54000000         | 5668096     | 65                | 34    | 33                        | 37            | 23         |
| Inception_5c | 66358384         | 71000000         | 4641616     | 67                | 36    | 34                        | 39            | 23         |




Estimated Result from synthesized kernels on FPGA using Global Memory

|              |                  |              |           |                 |           |                     |          | 
|--------------|------------------|--------------|-----------|-----------------|-----------|---------------------|----------| 
| Layer Name   | Total Operations | Total Cycles | Fmax      | Total Exec Time| Ops/cycle | Global Memory (r/w) | Ops/byte | 
| inception_3b | 599302144        | 689238       | 188.3 MHz | 3.66 ms    | 869       | 37.91 MB            | 15       | 
| inception_3c | 265732096        | 534396       | 215.6 MHz | 2.48 ms    | 497.25    | 11.15 MB            | 22.72    | 
| inception_4b | 71199744         | 146735       | 195.8 MHz | 0.752 ms    | 485.22    | 6.65 MB             | 10.2     | 
| inception_4c | 83091456         | 152837       | 180.0 MHz | 0.849 ms    | 543       | 5.26 MB             | 15       | 
| inception_4d | 95033344         | 157077       | 211.1 MHz | 0.744 ms    | 605       | 5.52 MB             | 16.4     | 
| inception_4e | 112093184        | 166768       | 210.0 MHZ | 0.794 ms    | 672.15    | 5.92 MB             | 18       | 
| inception_4f | 157151232        | 231294       | 206.0 MHz | 1.12 ms    | 649       | 7.64 MB            | 19.59    | 
| inception_5b | 47867904         | 82995        | 212.0 MHz | 0.391 ms    | 576       | 5.97 MB             | 7.63     | 
| inception_5c | 66906112         | 110384       | 198.0 MHz | 0.550 ms    | 606       | 10.84 MB            | 5.81     | 
