kernel_name_inception3b = inception_3b
kernel_name_inception3c = inception_3c
kernel_name_inception4b = inception_4b
kernel_name_inception4c = inception_4c
kernel_name_inception4d = inception_4d
kernel_name_inception4e = inception_4e
kernel_name_inception4f = inception_4f
kernel_name_inception5b = inception_5b
kernel_name_inception5c = inception_5c
design_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/googlenet_bitstreams
board_name = p520_max_sg280l
						 
inception_3b :               
	aoc -v -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception0.aocx  ./$(kernel_name_inception3b).cl
	
inception_3c :
	aoc -v -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception1.aocx  ./$(kernel_name_inception3c).cl

inception_4b :
	aoc -v -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception2.aocx  ./$(kernel_name_inception4b).cl

inception_4c :
	aoc -v -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception3.aocx  ./$(kernel_name_inception4c).cl
	
inception_4d :
	aoc -v -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception4.aocx  ./$(kernel_name_inception4d).cl
	
inception_4e :
	aoc -v -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception5.aocx  ./$(kernel_name_inception4e).cl
	
inception_4f :
	aoc -v -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception6.aocx  ./$(kernel_name_inception4f).cl
	
inception_5b :
	aoc -v -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception7.aocx  ./$(kernel_name_inception5b).cl
	
inception_5c :
	aoc -v -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception8.aocx  ./$(kernel_name_inception5c).cl
	
dummy :
	echo "Here"