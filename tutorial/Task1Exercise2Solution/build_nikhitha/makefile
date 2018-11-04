all : mark SimpleKernel.aocx

VERSION := 17.1
BOARD := p385a_sch_ax115

mark : main.o utility.o 
	g++ -L/opt/intelFPGA_pro/$(VERSION)/hld/host/linux64/lib -lOpenCL -o mark main.o utility.o
main.o : main.cpp utility.h
	g++ -I/opt/intelFPGA_pro/$(VERSION)//hld/host/include -c main.cpp
utility.o : utility.cpp
	g++ -I/opt/intelFPGA_pro/$(VERSION)//hld/host/include -c utility.cpp
SimpleKernel.aocx : SimpleKernel.cl
	aoc -march=emulator -board=$(BOARD) SimpleKernel.cl
clean:	
	rm -r main.o utility.o mark SimpleKernel.aocx
