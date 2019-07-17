test_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs/test_bitstreams
board_name = p520_max_sg280l

io_report: io_report_3b io_report_3c io_report_4b io_report_4c io_report_4d io_report_4e io_report_4f io_report_5b io_report_5c
	
io_report_3b:
	aoc  -v -rtl -report -board=$(board_name)   ./GoogleNet_Kernels_io_channels_inception0.cl
io_report_3c :
	aoc  -v -rtl -report -board=$(board_name)   ./GoogleNet_Kernels_io_channels_inception1.cl
io_report_4b :
	aoc  -v -rtl -report -board=$(board_name)   ./GoogleNet_Kernels_io_channels_inception2.cl
io_report_4c :
	aoc  -v -rtl -report -board=$(board_name)   ./GoogleNet_Kernels_io_channels_inception3.cl
io_report_4d :
	aoc  -v -rtl -report -board=$(board_name)   ./GoogleNet_Kernels_io_channels_inception4.cl
io_report_4e :
	aoc  -v -rtl -report -board=$(board_name)   ./GoogleNet_Kernels_io_channels_inception5.cl
io_report_4f :
	aoc  -v -rtl -report -board=$(board_name)   ./GoogleNet_Kernels_io_channels_inception6.cl
io_report_5b :
	aoc  -v -rtl -report -board=$(board_name)   ./GoogleNet_Kernels_io_channels_inception7.cl
io_report_5c :
	aoc  -v -rtl -report -board=$(board_name)   ./GoogleNet_Kernels_io_channels_inception8.cl
io_emulation :
	aoc  -v -march=emulator -board=$(board_name) -o $(test_dir)/GooleNet_Kernels_IO.aocx   ./GooleNet_Kernels_IO.cl
