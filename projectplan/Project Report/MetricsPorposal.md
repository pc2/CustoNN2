# Metrics

To measure the performance of our solution, we propose the following metrics 
- FLOPS
- Operations per cycle
- Latency
- Throughput
- Accuracy
- Rate Correct Score

### FLOPS

Floating point operations per second (FLOPS) is a  unit of measure for the numerical computing performance of a computer. FLOPS is technically the right term to use when the data being worked upon is of floating point type.  
In our case , with the quantization of weights from floating points  to integer type , we may have to refer to this unit of measurement as Integer Operations per Second (IOPS).
But the computing  industry and/or academia refers to “just integer” operations as MIPS (Millions of instructions per second). So depending on the eventual nature of the weights and the operational units on the FPGA, 
we have to decide whether to report this performance measure in FLOPS or MIPS.  
To calculate FLOPS (and/or MIPS) , we look at the source code and the final profiler report from the executed code. 
From the source code, we calculate the number of operations sans the overheads (loop index calculations, loop increment operations etc). And from the profiler report, we get a concrete idea of the clock frequency of the FPGA and the run time – the execution time. Combining these two allows us to calculate  FLOPS/MIPS. 

### Operations per cycle
FLOPS/MIPS are quantities which are dependent on the clock frequencies supported by the FPGA boards. 
When we want to just showcase the performance decoupled from the clock frequencies , we make use of Operations per cycle.  
The way to measure Operations per cycle is similar to the way we find FLOPS.

### Latency
Latency is generally defined as the time delay between initial input to a system and the output from the system.  
Latency captures the time it takes to load  data ,preprocess it, send said data over a network to the Inference Engine ,time needed for inference , and time needed to send the classified data back to the user. 
Latencies can be measured for just a single image or over a batch of images. We can expect to have different latencies based on how we measure it(single image vs batch of images).
[{1}][1]

### Throughtput
Throughput is  a measure of amount of information processed per unit time. In our case , we define it as the number of bytes processed per second.  
There are well known techniques to improve throughtput like batch processing where a number of images is batched together and sent for inference job. 
But as the batch size goes up , latencies also tend to go up. So there is a trade off involved here between through put and latency.[{1}][1]


### Accuracy
Accuracy is defined as the fraction of the number of correct inferences made to the total number of inferences made.  
In our context, accuracy depends on CNN topology used , nature of weights , training data etc. 


### Rate Correct Score
Taking inspiration from the field of Memory and  Cognitive research , we can combine accuracy and latency by considering Rate Correct Score[{2}}][2]   
It is defined as   

RCS= c/(∑Rt)

where c is the accuracy and Rt is the latency. 






## References
[[1]] J Hanhirova et al , Latency and Throughput Characterization of Convolutional Neural Networks for Mobile Computer Vision   
[[2]] André Vandierendonck , A comparison of methods to combine speed and accuracy measures of performance: A rejoinder on the binning procedure
  
[1]: https://arxiv.org/pdf/1803.09492.pdf
[2]: https://rdcu.be/brvnb
