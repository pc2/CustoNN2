test_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/aayushb/test_bitstreams
board_name = p520_max_sg280l

io_emulation :
	aoc  -v -march=emulator -board=$(board_name) -o $(test_dir)/GoogleNet_Kernels_IO.aocx   ./GoogleNet_Kernels_IO.cl
	aoc  -v -march=emulator -board=$(board_name) -o $(test_dir)/GoogleNet_Kernels_IO_1.aocx   ./GoogleNet_Kernels_IO_1.cl
emulate_kernels:
	aoc  -v -march=emulator -board=$(board_name) -emulator-channel-depth-model=default -o $(test_dir)/GoogleNet_Kernels_IO.aocx   ./GoogleNet_Kernels_channels.cl
	aoc  -v -march=emulator -board=$(board_name) -emulator-channel-depth-model=default -o $(test_dir)/GoogleNet_Kernels_IO_1.aocx   ./GoogleNet_Kernels_channels_1.cl

dynchan:
	aoc  -v -march=emulator -board=$(board_name) -emulator-channel-depth-model=default -o $(test_dir)/GoogleNet_Kernels_dyn.aocx   ./GoogleNet_Kernels_channels_dynamic.cl
	aoc  -v -march=emulator -board=$(board_name) -emulator-channel-depth-model=default -o $(test_dir)/GoogleNet_Kernels_dyn_1.aocx   ./GoogleNet_Kernels_channels_dynamic_1.cl	

global:
	aoc  -v -march=emulator -board=$(board_name) -emulator-channel-depth-model=default -o $(test_dir)/GoogleNet_Kernels.aocx   ./GoogleNet_Kernels_global.cl