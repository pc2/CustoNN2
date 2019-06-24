kernel_name = inception_modified_nnvm
design_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/inception_modified_nnvm
board_name = p520_max_sg280l

kernel_emulator :
	aoc  -v -march=emulator -board=$(board_name)   -o $(design_dir)/$(kernel_name).aocx  ./$(kernel_name).cl

report :
	aoc -rtl -report -board=$(board_name) ./$(kernel_name).cl