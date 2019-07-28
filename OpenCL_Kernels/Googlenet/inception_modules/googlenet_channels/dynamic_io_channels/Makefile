.PHONY: clean
kernel_name = GoogleNet_Kernels_channel_1a-3a
design_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2
board_name = p520_max_sg280l

 

dynchan:
	aoc  -v -march=emulator -board=$(board_name)   -o ./temp/GoogleNet_Kernels.aocx  ./$(kernel_name).cl

clean:
	rm -r ./temp