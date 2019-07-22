test_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs/test_bitstreams
board_name = p520_max_sg280l

io_emulation :
	aoc  -v -march=emulator -board=$(board_name) -o $(test_dir)/GoogleNet_Kernels_IO.aocx   ./GoogleNet_Kernels_IO.cl
	aoc  -v -march=emulator -board=$(board_name) -o $(test_dir)/GoogleNet_Kernels_IO_1.aocx   ./GoogleNet_Kernels_IO_1.cl
