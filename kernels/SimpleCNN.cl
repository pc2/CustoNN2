/*
 * Kernel for ConvolutionLayer.
 */

__kernel void ConvolutionLayer(__global double * restrict img,  
				__global double * restrict output, 
				__global double * restrict weights, 
				__global double * restrict bias,
				int number_of_filters, 
				int number_of_image_rows, 
				int number_of_image_cols, 
				int conv_stride){
	
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;

	index = 0;
	image_size = number_of_image_rows * number_of_image_cols;
	for (layer = 1; layer <=number_of_filters; layer++) {
		image_index = 0;
		for (i = 0; i < number_of_image_rows; i++) {
			for (j = 0; j < number_of_image_cols; j++) {
				current_layer = layer * 25;
				output[index] = bias[layer-1];
				filter_index = 0;
				for (k = - conv_stride; k <= conv_stride; k++) {
					for (t = - conv_stride; t <= conv_stride; t++) {
						temp_index = 28 * k + t + image_index;
						if ((j == 0 && t < 0) || (j == number_of_image_cols-1 && t > 0)) {
							temp_index = -1;
						}
						if ((j == 1 && t < -1) || (j == number_of_image_cols-2 && t > 1)) {
							temp_index = -1;
						}
						if ((i == 0 && k < 0) || (i == number_of_image_rows-1 && k > 0)) {
							temp_index = -1;
						}
						if ((i == 1 && k < -1) || (i == number_of_image_rows-2 && k > 1)) {
							temp_index = -1;
						}
						if ((temp_index >= 0) && (temp_index < image_size)) {
							output[index] += img[temp_index]*weights[filter_index+current_layer-25];
						}
						filter_index++;
					}
				}
				if (output[index] < (double)0)
					output[index] = (double)0;
				index++;
				image_index++;

			}
		}
	}	
}


/*
 * Kernel for Maxpooling Layer
 * Used for reducing the dimension of images. We use a 2*2 matrix for maxpooling.
 */

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


/*
 * Kernel for Fully Connected Layer.
 * Output : A single label for the digit it represents
 */

__kernel void FCL_Kernel(__global volatile int * restrict input,
				__global short *restrict weights,
		 		__global unsigned char *restrict output_labels, 
		 		int number_of_filters, 
		 		int weight_number)
		  	long maxScore=0;
		  	int weightIndex=0;
		  	int image_size = maxpool_out_rows * maxpool_out_cols * number_of_filters;

			for(int w = 0; w < weight_number; w++){
				long  sum=0;
				for(int j=0;j<image_size;j++){
					sum+= curr_image[j] * (int)weights[(w*image_size)+j];
				}
				if (sum > maxScore)
				{
	       		// Weight Having max score.
	        	maxScore = sum;
	        	weightIndex = w;
	            }

			}

			output_labels[i]=weightIndex;
}
