### Design : inception0
Fmax : 287.5MHz

| Kernel (inception0)                       |  No.of Operations (In Millions) |  Fraction of total Ops  |  Global Mem (r) (in MB)  |  Global Mem (w)(in MB)       |  Exe time(measured in ms) | Operations/second (measured) (in GOPS) |  Ops/cycle (measured)  | Ops/cycle (estimated)  | Channel Read(in MB) | Channel Write (in MB) |  Ops/byte  |  Global mem/second (in MBps) | Global mem/cycle (Bytes) |
|-------------------------------------------|---------------------------------|-------------------------|--------------------------|------------------------------|---------------------------|----------------------------------------|------------------------|------------------------|---------------------|-----------------------|------------|------------------------------|--------------------------|
| Padding_Conv2d_1a_7x7_Conv2D              | 0                               | 0                       | 0.588                    | 0                            | 1.79                      | 0                                      | 0                      | 0                      | 0                   | 0.614                 | 0          | 328.4916201                  | 2.045217391              |
| Conv2d_1a_7x7_Conv2D                      | 256                             | 0.23                    | 453                      | 0                            | 79.54                     | 3.21                                   | 15                     | 14                     | 0.614               | 3.21                  | 0.57       | 5695.247674                  | 1575.652174              |
| MaxPool_2a_3x3_MaxPool                    | 0                               | 0                       | 0                        | 0                            | 0.13                      | 0                                      | 0                      | 0                      | 3.21                | 0.802                 | 0          | 0                            | 0                        |
| Conv2d_2b_1x1_Conv2D                      | 27                              | 0.02                    | 50                       | 0                            | 17.16                     | 1.57                                   | 7.3                    | 8                      | 0.802               | 0.802                 | 0.54       | 2913.752914                  | 173.9130435              |
| Padding_Conv2d_2c_3x3_Conv2D              | 0                               | 0                       | 0                        | 0                            | 0.03                      | 0                                      | 0                      | 0                      | 0.802               | 0.841                 | 0          | 0                            | 0                        |
| Conv2d_2c_3x3_Conv2D                      | 848                             | 0.75                    | 1400                     | 0                            | 150                       | 5.653333333                            | 19.66376812            | 22                     | 0.841               | 2.4                   | 0.6        | 9333.333333                  | 4869.565217              |
| MaxPool_3a_3x3_MaxPool                    | 0                               | 0                       |  32bits                  | 0                            | 0.1                       | 0                                      | 0                      | 0                      | 2.4                 | 0.602                 | 0          | 40KB                         | 0                        |
| Total                                     | 1130                            | 1                       | 1903.588                 | 0                            | 252                       | 4.484126984                            | 8.5                    | 9.38                   | 7.86                | 7.86                  | 0.59       | 7553.920635                  | 6621.175652              |


### Design : inception2
Fmax : 285MHz

| Kernel (inception3)                            |  No.of Operations (In Millions) |  Fraction of total Ops  |  Global Mem (r) (in MB)  |  Global Mem (w)(in MB)       |  Exe time(measured in ms) | Operations/second (measured) (in GOPS) |  Ops/cycle (measured)  | Ops/cycle (estimated)  | Channel Read(in MB) | Channel Write (in MB) |  Ops/byte  |  Global mem/second (in MBps) | Global mem/cycle (Bytes) |
|------------------------------------------------|---------------------------------|-------------------------|--------------------------|------------------------------|---------------------------|----------------------------------------|------------------------|------------------------|---------------------|-----------------------|------------|------------------------------|--------------------------|
| feeder_3c                                      | 0                               | 0                       | 4B                       | 0                            | 4                         | 0                                      | 0                      | 0                      | 0.776               | 0.776                 | 0          | 0                            | 0                        |
| Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D         | 51.5                            | 0.08                    | 0.5                      | 0.38                         | 116                       | 0.443965517                            | 1.557773745            | 4                      | 0.776               | 0                     | 103        | 4.310344828                  | 1.765880218              |
| Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D         | 51.48                           | 0.08                    | 0.5                      | 0                            | 90                        | 0.572                                  | 2.007017544            | 2                      | 0.776               | 0.392                 | 103        | 5.555555556                  | 1.754385965              |
| Padding_Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D | 0                               | 0                       | 0                        | 0                            | 1.27                      | 0                                      | 0                      | 0                      | 0.392               | 0.45                  | 0          | 0                            | 0                        |
| Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D         | 424                             | 0.69                    | 0.79                     | 0.57                         | 60                        | 7.066666667                            | 24.79532164            | 7                      | 0.45                | 0                     | 311        | 13.16666667                  | 2.805263158              |
| Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D         | 12                              | 0                       | 0.12                     | 0                            | 29                        | 0.413793103                            | 1.451905626            | 4                      | 0.776               | 0.098                 | 92         | 4.137931034                  | 0.421052632              |
| Padding_Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D | 0                               | 0.02                    | 0                        | 0                            | 1.27                      | 0                                      | 0                      | 0                      | 0.098               | 0.1125                | 0          | 0                            | 0                        |
| Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D         | 53                              | 0.09                    | 0.39                     | 0.28                         | 33                        | 1.606060606                            | 5.635300372            | 7                      | 0.1125              | 0                     | 79         | 11.81818182                  | 1.398192451              |
| Mixed_3c_Branch_3_MaxPool_0a_3x3_MaxPool       | 0                               | 0                       | 0                        | 0                            | 7                         | 0                                      | 0                      | 0                      | 0.776               | 0.194                 | 0          | 0                            | 0                        |
| Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D         | 25.75                           | 0.04                    | 0.25                     | 0.19                         | 57                        | 0.451754386                            | 1.585103109            | 2                      | 0.194               | 0                     | 58.5       | 4.385964912                  | 0.888888889              |
| Mixed_3c_concat                                | 0                               | 0                       | 1.42                     | 0                            | 1.87                      | 0                                      | 0                      | 0                      | 0                   | 1.43                  | 0          | 0                            | 0                        |
| Total                                          | 617                             | 1                       | 2.84                     | 1.42                         | 512                       | 1.205078125                            | 4.228344298            | 4.97                   | x                   | x                     | 217        | 5.546875                     | 9.97464364               |



### Overall 
Fmax avg : 230MHz

| No.of Operations(in Millions) |  Global Mem  (in MB) |  Exe time(measured in ms) | Operations/second (measured) (GOPS) |  Ops/cycle (measured)  |  Ops/byte  |  Global mem/second (GBps) |
|-------------------------------|----------------------|---------------------------|-------------------------------------|------------------------|------------|---------------------------|
|                               |                      |                           |                                     |                        |            |                           |
| 3298.24                       | 1971                 | 1180                      | 2.79                                | 12.15                  | 1.67       | 1.5                       |




## Improvement over unoptimized hybrid design :

Execution speed up = (1770ms/1180ms) = 1.5    
Throughput improvement = 2.79GOPS - 1.86GOPS = 0.93GOPS  
Latency improvement = 1770ms - 1180ms = 590ms  
Accuracy : Unchanged




