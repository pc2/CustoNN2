Task_2: main.o utility.o
	g++ -I/opt/intelFPGA_pro/17.1/hld/host/linux64/lib -lOpenCL -o Task2 main.o utility.o

main.o: main.cpp utility.h
	g++ -I/opt/intelFPGA_pro/17.1/hld/host/include -c main.cpp

utility.o: utility.cpp
	g++ -I/opt/intelFPGA_pro/17.1/hld/host/include -c utility.cpp

SimpleKernel.aocx: SimpleKernel.cl
	aoc -march=emulator -board=p385a_sch_ax115 SimpleKernel.cl

build_emulator_binary: 
	aoc -c SimpleKernel.cl

generate_report:
	aoc -report -c SimpleKernel.cl

clean: rm -r main.o utility.o Task SimpleKernel.aocx
