block1 = block1
block2 = block2
block3_1 = block3_1
block3_2 = block3_2
block4 = block4
design_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/resnet_emulation
board_name = p520_max_sg280l



block1 :
	aoc  -v -march=emulator -board=$(board_name)   -o $(block1).aocx  ./$(block1).cl

report_block1 :
	aoc -rtl -report -board=$(board_name) ./$(block1).cl

block2 :
	aoc  -v -march=emulator -board=$(board_name)   -o $(block2).aocx  ./$(block2).cl

report_block2:
	aoc -rtl -report -board=$(board_name) ./$(block2).cl


block3_1 :
	aoc  -v -march=emulator -board=$(board_name)   -o $(block3_1).aocx  ./$(block3_1).cl

report_block3_1 :
	aoc -rtl -report -board=$(board_name) ./$(block3_1).cl


block3_2 :
	aoc  -v -march=emulator -board=$(board_name)   -o $(block3_2).aocx  ./$(block3_2).cl

report_block3_2 :
	aoc -rtl -report -board=$(board_name) ./$(block3_2).cl


block4 :
	aoc  -v -march=emulator -board=$(board_name)   -o $(block4).aocx  ./$(block4).cl

report_block4 :
	aoc -rtl -report -board=$(board_name) ./$(block4).cl


