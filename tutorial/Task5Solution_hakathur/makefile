.PRECIOUS: summation.aocx summation_%.aocx summation.aoco summation_%.aoco

summation.aocx: summation.cl
	aoc -march=emulator summation.cl
	-rm summation.aoco

summation_%.aocx: summation-%.cl
	aoc -march=emulator summation-$*.cl
	-rm summation_$*.aoco

summation.aoco: summation.cl
	aoc -c -report summation.cl

summation_%.aoco: summation-%.cl
	aoc -c -report summation-$*.cl

buildAltera: summation.aocx
	@echo ""

buildAltera-%: summation_%.aocx
	@echo ""

report: summation.aoco
	@echo ""

report-%: summation_%.aoco
	@echo ""

summationHost: main.cpp utility.cpp utility.h
	g++ -fPIC -g main.cpp utility.cpp `aocl compile-config` `aocl link-config` -o summationHost

run: summationHost summation.aocx
	CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./summationHost

run-%: summationHost summation_%.aocx
	ln -sf summation_$*.aocx summation.aocx
	CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./summationHost

clean:
	-rm *.aoco
	-rm *.aocx
	-rm summationHost