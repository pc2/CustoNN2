# First generate an <file>.aoco file
build_emulator_binary: 
	aoc -c mykernels.cl

# Now compile the project to generate reports and to generate an <file>.aocr file
generate_reports:
	aoc -rtl mykernels.c

# Now compile the project into a hardware programming file <file>.aocx.
execute_host_code:
	aoc mykernels.aoco

all: build_emulator_binary generate_reports execute_host_code

clean:
	rm -r mykernels
	rm mykernels.aoco
	rm mykernels.aocr
