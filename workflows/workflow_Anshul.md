## Setting For Connections
- Using SSH and xRDP on Remmina-Ubuntu 18.04 to connect to CC Front-End.
- The performance via Eduroam is good , via home using Open-Vpn a little bit sluggish.

## Repositories
- A single copy is cloned in the working directory on the CC Frontend.

## Local copies of documentation
- I would frequently require the Git documentation , for that saved all the required webiste links in the pdf on the local machine.

## Activities of Task 2
- We will be working on the Custom Computing Cluster.(CC FrontEnd)
- This is currently used for research purpose on FPGA's.
- I expect that we will be using the following FPGA Boards -
  1) Nallatech 385A board with Intel/Altera Arria 10 GX 1150 FPGA.
  2) Alpha Data 7v3 board with Xilinx Virtex-7 XC7VX690T FPGA.
- And the following Drivers - 
  1) Intel/Altera 17.1.2
  2) Xilinx SDx 2017.

## Activities of Task 3
- Able to conect to the CC Front End Cluster through both SSH and xRDP.
- Sanity checks performed on Intel/Altera FPGA SDK for OpenCL.
- User is mounted to the following path - **/upb/departments/pc2/users/a/anshul**
- I am able to access and change the content of .bashrc file.

## TASK 4 

## Documentation

## Channels
- Uses Fifo mechanism for the buffers.
- It is used to communicate between the kernels and kernels and I/O devices
- It has a by default blocking behaviour but non blocking channels can also be implemented.
- Usin the Intel FPGA SDK pro for implementation of Channels

## Merge Request
- Merge Request can directly be made from the GitLab website. I haven't created any merge request uptil now.

## aoc Command Line options
- aoc --help gives documention regarding aoc command line options
    - aoc -version gives version name
    - aoc -list-boards lists out all the available boards.
    - aoc board=<board name> compiles for the specified boards.

## FPGA and SDK
- We are using the 17.1.2 and 18.0.1 version for the OpenCL SDK
- For Documentation referring to the PC2 wiki website(link -  https://wiki.pc2.uni-paderborn.de/pages/viewpage.action?pageId=19562930)

