kernel_name = resnet_updated
design_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/resnet_emulation
board_name = p520_max_sg280l

kernel_emulator :
	aoc  -v -march=emulator -board=$(board_name)   -o $(design_dir)/$(kernel_name).aocx  ./$(kernel_name).cl

report :
	aoc -rtl -report -board=$(board_name) ./$(kernel_name).cl

googlenet:
	aoc  -v -march=emulator -board=$(board_name)   -o $(design_dir)/GoogleNet_Kernels.aocx  ./GoogleNet_Kernels.cl

resnet:
	aoc  -v -march=emulator -board=$(board_name)   -o $(design_dir)/ResNet_Kernels.aocx  ./$(kernel_name).cl
