/*
 * Kernel for ConvolutionLayer.
 */

__kernel void ConvolutionLayer(__global unsigned char * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				int number_of_filter_rows, 
				int number_of_filter_cols,
				int number_of_filters, 	
				int number_of_images,		 
				int number_of_image_rows, 
				int number_of_image_cols, 
				int conv_pad,
				int stride,
				__global double * restrict output){
	
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;

	index = 0;
	image_size = number_of_image_rows * number_of_image_cols;
	for (image_number = 0; image_number < number_of_images; image_number++){
	for (layer = 1; layer <=number_of_filters; layer++) {
		image_index = number_of_image_rows * number_of_image_cols * image_number;
		for (i = 0; i < number_of_image_rows; i++) {
			for (j = 0; j < number_of_image_cols; j++) {
				current_layer = layer * 25;
				output[index] = bias[layer-1];

				filter_index = 0;
				for (k = - conv_pad; k <= conv_pad; k++) {
					for (t = - conv_pad; t <= conv_pad; t++) {
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
						if ((temp_index >= 0) && (temp_index < image_size + image_index)) {
							output[index] += img[temp_index]*weights[filter_index+current_layer-25];
						}
						filter_index++;
					}
				}
				//Relu
				if (output[index] < 0)
					output[index] = 0;

				//if((image_number==0||image_number==1) && layer==1)
                //	printf("%f  ",output[index]);

				index++;
				image_index++;

			}
			//if((image_number==0||image_number==1) && layer==1)
            //	printf("\n");
		}
		//if((image_number==0||image_number==1) && layer==1)
        //	printf("\n\n");

		}
	}
}


/*
 * Kernel for Maxpooling Layer
 * Used for reducing the dimension of images. We use a 2*2 matrix for maxpooling.
 */

__kernel void MaxPool(__global double * restrict input, 
			int number_of_image_rows, 
			int  number_of_image_cols,
			int number_of_filters, 
			int stride,
			int number_of_images,
			__global double * restrict output){
	int count = 0;
	double maxpool[4]; //maxpooing matrix 1D matrix with 0 and 1 position has r1c1 r1c2 and 2 and 3 has r2c1 r2c2
	for ( int i =0; i < number_of_images; ++i)
        {
            
            for (int k = 0; k <number_of_filters; ++k)
            {
                for (int x = 0; x < number_of_image_rows; x=x+stride)
                {
                    for (int y = 0; y < number_of_image_cols; y=y+stride)
                    {
						double max=0.0;
                        maxpool[0] = input[(i*number_of_image_rows*number_of_image_cols*number_of_filters)+(k*number_of_image_rows*number_of_image_cols)+(x*number_of_image_cols)+(y)];
                        maxpool[1] = input[(i*number_of_image_rows*number_of_image_cols*number_of_filters)+(k*number_of_image_rows*number_of_image_cols)+(x*number_of_image_cols)+(y+1)];
                        maxpool[2] = input[(i*number_of_image_rows*number_of_image_cols*number_of_filters)+(k*number_of_image_rows*number_of_image_cols)+(x*number_of_image_cols)+(y+number_of_image_cols)];
                        maxpool[3] = input[(i*number_of_image_rows*number_of_image_cols*number_of_filters)+(k*number_of_image_rows*number_of_image_cols)+(x*number_of_image_cols)+(y+number_of_image_cols+1)];
                                        
						max = maxpool[0];
                        for(int j = 1; j<4; j++)
						{		
        					if(maxpool[j] > max)
        						max = maxpool[j];
						}
					
						output[count] = max;
						//if((i==1 || i==0) && k==0)
                        //	printf("%f  ",output[count]);
						count++;
										

                    }

                	//if((i==1 || i==0) && k==0)
                    //   printf("\n ");
                    
				   }
        	        //if((i==1 || i==0) && k==0)
                    //printf("\n\n\n ");
                }

        }

};


/*
 * Kernel for Fully Connected Layer.
 * Output : A single label for the digit it represents
 */

__kernel void FCL_Kernel(__global double * restrict input,
				__global float * restrict weights,
				__global float * restrict bias,
				int number_of_pixels,
				int weight_number,
				int number_of_images,
		 		__global int * restrict output_labels, 
				int number_of_image_rows, 
				int  number_of_image_cols,
		 		int number_of_filters)
{
	
	int image_number;
	int image_size = (number_of_image_rows) * (number_of_image_cols) * number_of_filters;
	for (image_number = 0; image_number < number_of_images; image_number++){
		double maxScore=0.0;
		int weightIndex=0;
		
		int curr_pos = image_number * image_size;
		//double  sum=0.0;
		for(int w = 0; w < weight_number; w++){
			double sum=bias[w];
			for(int j=0;j<image_size;j++){
				sum+= input[curr_pos+j] * weights[(w*image_size)+j];
				if(w==0)
					maxScore=sum;
			}
			//if(image_number<3)
            	//printf("%d Image %d -- %f -- %f\n  ",image_number,w,sum,bias[w]);

			if (sum >= maxScore){
		       	// Weight Having max score.
				maxScore = sum;
				weightIndex = w;
		    	}
			sum=0.0;
		}

		output_labels[image_number]=weightIndex;
		//if(image_number<100)
			//printf(" Classification for 1st Image : %d \n",weightIndex);
	}
}
