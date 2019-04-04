#define G_NUMBER_OF_IMAGES 10000
#define G_NUMBER_OF_FILTERS 32
#define G_MAXPOOL_OUT_ROWS 14
#define G_MAXPOOL_OUT_COLS 14
#define G_WEIGHT_NUMBER 10

__kernel void FCL(__global volatile int * restrict input,
				__global short *restrict weights,
		 		__global unsigned char *restrict output_labels)
{				
		  	long maxScore=0;
		  	int weightIndex=0;
		  	int image_size = G_MAXPOOL_OUT_ROWS * G_MAXPOOL_OUT_COLS * G_NUMBER_OF_FILTERS;

			for(int w = 0; w < G_WEIGHT_NUMBER; w++){
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
