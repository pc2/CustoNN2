#### Which parts of the infrastructure do you expect to use within the project?
- Custom Computing (CC) Cluster
	- CC Frontend: Host Code and OpenCL development, emulation, synthesis
	- cc-7: Nallatech 385A board with Intel/Altera Arria 10 GX 1150 FPGA
- Noctua HPC Cluster
	- 16 FPGA nodes each with 2 Nallatech 520N boards with Stratix 10 FPGAs
	- Individual compute nodes or FPGA nodes for synthesis
- HPC Cluster Oculus [https://pc2.uni-paderborn.de/hpc-services/available-systems/oculus/](https://pc2.uni-paderborn.de/hpc-services/available-systems/oculus/)
	- GPU nodes for CNN training

#### Which tools, FPGAs and boards do you expect to use within the project?
- Intel FPGA SDK for OpenCL
	- 17.1 for Arria 10
	- 18.0.1 for Stratix 10 (currently)
	- 18.1 for Stratix 10 (soon)
	- 19.0 for Stratix 10 (next year)

#### Which of the sanity checks from the FPGA documentation can you perform this machine, does it contain FPGAs?
- The machine does not contain FPGAs, but the tools are fully installed, so all steps that don't require FPGA hardware can be performed. From the wiki documentation these are
	- `aoc -version`
	- `quartus_cmd -version`
	- `aoc -list-boards` This lists the boards for which you can compile.
- From the first programming task
	- `aoc -march=emulator -board=p385a_sch_ax115 SimpleKernel.cl` (Compile for emulation)
	- `CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 hostbinary SimpleKernel.aocx` (Perform software emulation)

#### What is the path to your mounted user home on this system? What is your quota here (use `df -h` to find out)?
- typically 5GB

#### Can you access and edit the .bashrc file there?
- This is the place where you can automatically apply the fix to gitlab problems with xrdp and also provide shortcuts to load specific tool versions.

#### In the file system, go to /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2 and setup a local working directory with your IMT username. What is your quota here (use again `df -h` to find out)?
- 15TB shared Quota for everyone in the user group