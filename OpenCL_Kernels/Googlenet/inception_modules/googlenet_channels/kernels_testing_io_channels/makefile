test_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs/test_bitstreams
board_name = p520_max_sg280l

io_emulation :
	aoc  -v -march=emulator -board=$(board_name) -o $(test_dir)/GoogleNet_Kernels_IO.aocx   ./GoogleNet_Kernels_IO.cl
	aoc  -v -march=emulator -board=$(board_name) -o $(test_dir)/GoogleNet_Kernels_IO_1.aocx   ./GoogleNet_Kernels_IO_1.cl
emulate_kernels:
	aoc  -v -march=emulator -board=$(board_name) -emulator-channel-depth-model=default -o $(test_dir)/GoogleNet_Kernels_IO.aocx   ./GoogleNet_Kernels_channels.cl
	aoc  -v -march=emulator -board=$(board_name) -emulator-channel-depth-model=default -o $(test_dir)/GoogleNet_Kernels_IO_1.aocx   ./GoogleNet_Kernels_channels_1.cl
	aoc  -v -march=emulator -board=$(board_name) -emulator-channel-depth-model=default -o $(test_dir)/GoogleNet_Kernels_IO_2.aocx   ./GoogleNet_Kernels_channels_2.cl
