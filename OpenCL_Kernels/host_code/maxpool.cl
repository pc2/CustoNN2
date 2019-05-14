__kernel void MaxPool(__global int * restrict input, 
			int number_of_image_rows, 
			int  number_of_image_cols,
			int number_of_filters, 
			int stride,
			int padding,
			int number_of_images,
			__global int * restrict output){

	int count = 0,modVal = number_of_image_cols - stride; //mod val is used to skip certain loops
	double maxpool[9]; //maxpooing matrix 1D matrix with 0 and 1 position has r1c1 r1c2 and 2 and 3 has r2c1 r2c2
	int image_size = number_of_image_rows * number_of_image_cols * number_of_filters * number_of_images;
	int max;
	for(int i = 0; i<image_size;i+=stride){
	
		maxpool[0] = input[i];
		maxpool[1] = input[i+1];
		maxpool[2] = input[i+2];
		maxpool[3] = input[i+number_of_image_cols];
		maxpool[4] = input[i+number_of_image_cols + 1];
		maxpool[5] = input[i+number_of_image_cols + 2];
		maxpool[6] = input[i+2*number_of_image_cols];
		maxpool[7] = input[i+2*number_of_image_cols+1];
		maxpool[8] = input[i+2*number_of_image_cols+2];


		if(i%modVal == 0 && i!=0){
			modVal+=number_of_image_cols * stride;
			i+=number_of_image_cols; //2 rows are considered at a time
		}
		max = maxpool[0];       
		for(int j = 1; j<9; j++)
			{
        		  if(maxpool[j] > max)
        		        max = maxpool[j];
			}
     	output[count] = max;
		count+=1; 

	}
};

