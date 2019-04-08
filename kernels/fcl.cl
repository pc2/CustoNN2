__kernel void FCL(__global volatile int * restrict input,
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

