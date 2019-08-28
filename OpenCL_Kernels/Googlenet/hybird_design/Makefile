.PHONY: clean

ker=GoogleNet_Hybrid_4b.cl
board_name = p520_max_sg280l
check:
	aoc  -v -march=emulator -board=$(board_name)   -o ./temp/GoogleNet_Kernels.aocx  ./$(ker)
	
clean:
	rm -r ./temp