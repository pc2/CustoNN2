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
				__global int * restrict conv_pad_begin,
				__global int * restrict conv_pad_end,				
				int stride,
				__global double * restrict output){
	

	printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	int num_rows_after_padding,num_cols_after_padding;
	num_rows_after_padding = number_of_image_rows + conv_pad_begin[0] + conv_pad_end[0];
	num_cols_after_padding = number_of_image_cols + conv_pad_begin[1] + conv_pad_end[1];
	__local unsigned char temp_image_padded[512 * 512];
	index = 0;
	image_size = number_of_image_rows * number_of_image_cols;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		//printf("Before padding\n");
		//padding logic
		for(int r=0,r_in=0;r<num_rows_after_padding && r_in<number_of_image_rows;r++)
		{
			for(int c=0,c_in=0;c<num_cols_after_padding && c_in<number_of_image_cols;c++)
			{
				if((r<conv_pad_begin[0]||r>(number_of_image_rows+1))||(c<conv_pad_begin[1]||c>(number_of_image_cols+1)))
				{
					temp_image_padded[(num_cols_after_padding*r)+c]=0.0;
				}
				else
				{
					temp_image_padded[(num_cols_after_padding*r)+c]=img[(image_number*number_of_image_rows*number_of_image_cols)+(number_of_image_cols*r_in)+c_in];
					c_in++;
				}

			}
			if(r>(conv_pad_begin[0]-1)&&r<(number_of_image_rows+1))
			{
				r_in++;
			}
		}

		//printf("Padding done\n");
		
		for (layer = 0; layer <number_of_filters; layer++) 
		{
			//image_index = number_of_image_rows * number_of_image_cols * image_number;
			for (i = 0; i < number_of_image_rows; i++) 
			{
				for (j = 0; j < number_of_image_cols; j++) 
				{
				
					double temp_conv_val = bias[layer];
					int PaddedX = i;
					int PaddedY = j;

					 for(int filterX=0; filterX<number_of_filter_rows; filterX++)
			      		 {
			    	  		 for(int filterY=0; filterY<number_of_filter_cols; filterY++)
			    	  		 {
			               			 temp_conv_val  += (int)temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY] *weights[(layer*number_of_filter_rows*number_of_filter_cols)+(filterX*number_of_filter_cols)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                		PaddedY++;
			           		 }
			           		PaddedX++;
			           		PaddedY=j;
					}

					
					output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
					index++;
				


				

				}
		
			}
	

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
