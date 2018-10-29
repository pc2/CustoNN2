## Questions to answer for the allocated OpenCL kernel file
- Is the kernel a single work item kernel or an ND Range kernel?
- Which task does the kernel perform?
- Which input arguments does the kernel require, how are they prepared on the host side?
- If it is an ND Range kernel, what global size and local size can you supply or do you have to supply?
- Which memory address space do variables/arrays occupy, that don't have an explicit qualifier (`__global`, `__local`, `__constant`, or `__private`)? Would you suggest any changes to the address spaces?