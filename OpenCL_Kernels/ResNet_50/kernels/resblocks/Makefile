kernel_name_block1 = block1
kernel_name_block2 = block2
kernel_name_block3_1 = block3_1
kernel_name_block3_2 = block3_2
kernel_name_block4 = block4

design_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/resnet_emulation
board_name = p520_max_sg280l
						 
block1 :               
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/block1.aocx  ./$(kernel_name_block1).cl
	
block2 :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/block2.aocx  ./$(kernel_name_block2).cl

block3_1 :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/block3_1.aocx  ./$(kernel_name_block3_1).cl

block3_2 :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/block3_2.aocx  ./$(kernel_name_block3_2).cl
	
block4 :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/block4.aocx  ./$(kernel_name_block4).cl
	

