task2 : main.o utility.o 
	g++ -L/opt/intelFPGA_pro/17.1/hld/host/linux64/lib -lOpenCL -o task2 main.o utility.o 

main.o : main.cpp utility.h
	g++ -I/opt/intelFPGA_pro/17.1/hld/host/include -c main.cpp
utility.o : utility.cpp
	g++ -I/opt/intelFPGA_pro/17.1/hld/host/include -c utility.cpp
SimpleKernel.aocx : SimpleKernel.cl 
	aoc -march=emulator -board=p385a_sch_ax115 SimpleKernel.cl
clean : rm -r main.o utility.o
