## Routing Kernels 
- As we move towards running CNNs on more than 1 FPGA , we face the challenge of sending data from 1 FPGA to another in a pipelined fashion.
- The most effective way of moving data from the last kernel of one FPGA to the next would be using  external channels provided  by  Intel FPGA SDK .
- As per [Noctua FPGA  Wiki](https://wikis.uni-paderborn.de/pc2doc/Noctua-FPGA) , the FPGAs available to us are connected with each other in different sets of configurations.  
This means that we cannot have a repetitive way of writing our data to external channels.
For example : 
    Consider a design which requires us to use 4 FPGAs. Let FPGA1\`s output be read by FPGA2 and so on. 
To connect FPGA1 with FPGA2 , we may have to write to external channel0 (by consulting [Noctua FPGA  Wiki](https://wikis.uni-paderborn.de/pc2doc/Noctua-FPGA)) , and while trying to link FPGA3 with FPGA4 , we may have to write to channel1 (just to give an example).
So if we were to replicate our code at the last kernel which sits in an FPGA , we would run into trouble.
- Moreover , the "Load Balancer" assigns us nodes from a list of 16 nodes when we try to select a node (though we can use -w to select a particular node , but this would mean the end-user needs to have a detailed view of the node structure)
- So at compile time , we would not have the knowledge of nodes which would be assinged to us ,and thus no way of having writing to valid external channels.
- A solution to this would be to use  the ideas of [Routing Kernels](https://git.uni-paderborn.de/cs-hit/pg-custonn2-2018/blob/master/projectplan/presentation_slides/Routing_Kernels_ppt.pdf)  in the last kernel to write to appropriate external kernels based on information availabel at run time.
- As mentioned in the PDF linked above , we can easily write to different channels based on the argument passsed to the kernel.
- The argument passed to the routing+concat kernel (refering to our Googlenet implementation) needs to be determined during run-time. 
- By quering info about  nodes assinged to our exe job request  , we can pass the appropriate arguments to our last kernel at run-time.
- The details of this plan needs to be worked upon now.
- Without Routing Kernels , our design will be very static and would force us to hand pick  all the required nodes. 
