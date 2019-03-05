.PHONY : clean all run

AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)
AOCL_LINK_CONFIG_NALLATECH = -L/opt/intelFPGA_pro/17.1/hld/board/nalla_pcie/linux64/lib/ -lalteracl -lnalla_pcie_mmd -lelf

CC =g++ -std=c++11

#names of the cpp files
host_code=main
host_code_dependency =utility

kernel_name =Simple_ConvolutionNeuralNetwork
kernel_name_1 = Simple_ConvolutionNeuralNetwork_FC

#one directory above makefile`s directory - consider as  "parent directory"
current_dir = $(shell cd .. && pwd)

#directory where .cpp and .h files are placed
host_code_directory =$(current_dir)/src

#For Selecting the board
version = 17
ifeq  ($(version),17)
board_name = p385a_sch_ax115
else
$(Compiling for SDK version 18)
#one of the boards supported by the SDK 18.0.1
board_name = p520_max_sg280l
endif

#program execution
all: run

run: host_prog
	./host_prog


#creation of executable
host_prog : host_prog.o $(host_code_dependency).o 
	$(CC) -L/opt/intelFPGA_pro/17.1/hld/board/nalla_pcie/linux64/lib/ -L/opt/intelFPGA_pro/17.1/hld/host/linux64/lib -lalteracl -lnalla_pcie_mmd -lelf -o host_prog host_prog.o $(host_code_dependency).o
report : $(current_dir)/$(kernel_name).cl
	@echo "Generating Report for the Kernel - ${kernel_name} and ${kernel_name_1}"
	aoc -rtl -report -v  -board=$(board_name) $(current_dir)/$(kernel_name).cl
	


#below code generates all the rquired files - report gets generated when -board=emulator is removed. Generalize this ? Future task
kernel :  $(current_dir)/$(kernel_name).cl
	aoc  -march=emulator -board=$(board_name)  -emulator-channel-depth-model=ignore-depth -v -o $(current_dir)/$(kernel_name).aocx $(current_dir)/$(kernel_name).cl
	

host_prog.o : $(host_code_directory)/$(host_code).cpp
	$(CC) -c -o host_prog.o $(host_code_directory)/$(host_code).cpp $(AOCL_COMPILE_CONFIG)

utility.o : $(host_code_directory)/$(host_code_dependency).cpp
	$(CC) -c -o $(host_code_dependency).o $(host_code_directory)/$(host_code_dependency).cpp  $(AOCL_COMPILE_CONFIG)

synthesis : 
	aoc  -board=$(board_name)  -v  -o $(current_dir)/synthesis/$(kernel_name).aocx $(current_dir)/$(kernel_name).cl


#remove files created in build dir , parent dir and folder created in parent dir
clean :
	rm *.o ; rm $(current_dir)/*.aoc? ; rm -r $(current_dir)/$(kernel_name) ; rm host_prog ; rm *.aoco rm -r $(kernel_name)
