.PHONY : clean

AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)

#names of the cpp files
host_code=main_solution_ex2
host_code_dependency =utility

kernel_name =SimpleKernel

#one directory above makefile`s directory - consider as  "parent directory"
current_dir = $(shell cd .. && pwd)

#directory where .cpp and .h files are placed
host_code_directory =$(current_dir)/SimpleOpenCL


#hack to choose between SDK 17 and 18. Could be made more elegent and more general. Future task.
version = 17
ifeq  ($(version),17)
board_name = p385a_sch_ax115
else
$(Compiling for SDK version 18)
#one of the boards supported by the SDK 18.0.1
board_name = p520_max_sg280l
endif 	

#program execution
all: host_prog     
	host_prog

#creation of executable 
host_prog : host_prog.o $(host_code_dependency).o $(kernel_name).aocx
	g++ -o host_prog host_prog.o $(host_code_dependency).o $(AOCL_LINK_CONFIG)



#below code generates all the rquired files - report gets generated when -board=emulator is removed. Generalize this ? Future task
SimpleKernel.aocx :  $(current_dir)/$(kernel_name).cl
	aoc  -march=emulator -board=$(board_name) -profile -o $(current_dir)/$(kernel_name).aocx $(current_dir)/$(kernel_name).cl

host_prog.o : $(host_code_directory)/$(host_code).cpp
	g++ -c -o host_prog.o $(host_code_directory)/$(host_code).cpp $(AOCL_COMPILE_CONFIG)

utility.o : $(host_code_directory)/$(host_code_dependency).cpp
	g++ -c -o $(host_code_dependency).o $(host_code_directory)/$(host_code_dependency).cpp  $(AOCL_COMPILE_CONFIG)




#remove files created in build dir , parent dir and folder created in parent dir
clean :
	rm *.o ; rm $(current_dir)/*.aoc? ; rm -r $(current_dir)/$(kernel_name) ; rm host_prog ;





 
