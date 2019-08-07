.PHONY: clean
kernel_name_inception3a = GoogleNet_Kernels_channel_1a-3a
kernel_name_inception3b = GoogleNet_Kernels_channel_3b
kernel_name_inception3c = GoogleNet_Kernels_channel_3c
kernel_name_inception4b = GoogleNet_Kernels_channel_4b
kernel_name_inception4c = GoogleNet_Kernels_channel_4c
kernel_name_inception4d = GoogleNet_Kernels_channel_4d
kernel_name_inception4e = GoogleNet_Kernels_channel_4e
kernel_name_inception4f = GoogleNet_Kernels_channel_4f
kernel_name_inception5b = GoogleNet_Kernels_channel_5b
kernel_name_inception5c = GoogleNet_Kernels_channel_5c
#design_dir = /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/googlenet_channels_bitstreams
design_dir = ./
board_name = p520_max_sg280l

 

dynchan:
	aoc  -v -march=emulator -board=$(board_name)   -o ./temp/GoogleNet_Kernels.aocx  ./$(ker)
	
GoogleNet_Kernels_channel_1a-3a :               
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception0.aocx  ./$(kernel_name_inception3a).cl
	
GoogleNet_Kernels_channel_3b :               
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception1.aocx  ./$(kernel_name_inception3b).cl
	
GoogleNet_Kernels_channel_3c :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception2.aocx  ./$(kernel_name_inception3c).cl

GoogleNet_Kernels_channel_4b :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception3.aocx  ./$(kernel_name_inception4b).cl

GoogleNet_Kernels_channel_4c :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception4.aocx  ./$(kernel_name_inception4c).cl
	
GoogleNet_Kernels_channel_4d :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception5.aocx  ./$(kernel_name_inception4d).cl
	
GoogleNet_Kernels_channel_4e :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception6.aocx  ./$(kernel_name_inception4e).cl
	
GoogleNet_Kernels_channel_4f :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception7.aocx  ./$(kernel_name_inception4f).cl
	
GoogleNet_Kernels_channel_5b :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception8.aocx  ./$(kernel_name_inception5b).cl
	
GoogleNet_Kernels_channel_5c :
	aoc -v -profile -fp-relaxed -fpc -board=$(board_name) -o $(design_dir)/inception9.aocx  ./$(kernel_name_inception5c).cl

clean:
	rm -r ./temp