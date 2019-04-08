//max pooling
__kernel void MaxPool(__global double * restrict input, 
					__global double * restrict output, 
					int number_of_filters, 
					int number_of_image_rows, 
					int  number_of_image_cols){

	int count = 0,modVal = number_of_image_cols - 2; //mod val is used to skip certain loops
	double maxpool[4]; //maxpooing matrix 1D matrix with 0 and 1 position has r1c1 r1c2 and 2 and 3 has r2c1 r2c2
	int image_size = number_of_image_rows * number_of_image_cols * number_of_filters;
	int max;
	for(int i = 0; i<image_size;i+=2){
	
		maxpool[0] = input[i];
		maxpool[1] = input[i+1];
		maxpool[2] = input[i+number_of_image_cols + 1];
		maxpool[3] = input[i+number_of_image_cols + 2];

		if(i%modVal == 0 && i!=0){
			modVal+=number_of_image_cols * 2;
			i+=number_of_image_cols; //2 rows are considered at a time
		}
		max = maxpool[0];       
		for(int j = 1; j<4; j++)
			{
        		  if(maxpool[j] > max)
        		        max = maxpool[j];
			}
     	output[count] = max;
		count+=1; 

	}
};

