# CustoNN2

## ssh or xrdp for your normal workflow?
- I am okay with using either one of them.
- xrdp seems to be a little slow compared to ssh.
- For ssh, we need not use VPN when connecting from home.
- However for xrdp, we need VPN for accessing the Cluster when connecting from home.

## Cloned versions of the git repository
- One in the cluster in directory - /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs
- One more in the local machine.
- How to sync between repo : Using `git pull` command.

### Mount shared file system from the local system
Mounted the shared file system on my windows local machine using CIFS .
CIFS IMT Link : https://hilfe.uni-paderborn.de/Netzlaufwerk_einbinden_(Windows_7)

### Which documentation will you use frequently, how will you access it, do you need local copies of the relevant pdfs?
**TODO**

## TASK 2

### Which parts of the infrastructure do you expect to use within the project?
- Initially, we will be using the Custom Computing (CC) Cluster for development.
- After Development phase, we will be using HPC production Noctua Cluster.

### Which tools, FPGAs and boards do you expect to use within the project?
- We will be using **Nallatech 385A board with Intel/Altera Arria 10 GX 1150 FPGA** for development purpose.
- The Goal of this project is to deploy CNNs on **Nallatech 520N boards with Stratix 10 FPGA**.


## TASK 3

### Which of the sanity checks from the FPGA documentation can you perform this machine, does it contain FPGAs?
- `aoc -version` will display the compiler version.
- `quartus_cmd -version` will give us the version of the Quartus Prime.
- `aoc -list-boards` will list down the available boards connected to the machine.
- `aocl diagnose`  could not be executed since this check has to be done on FPGA Node.

### What is the path to your mounted user home on this system? What is  your quota here
- /upb/departments/pc2/users/a/adeshs
- Quota : 5GB

### Can you access and edit the .bashrc file there?
- Yes, .bashrc file is owned by me and I can edit it. We usually add environment variables inside .bashrc file. I added
`export PG_HOME_ADESH=/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs` as a variable for easy access of the project directory.
- Quota in Cluster is 15TB
