__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				int number_of_filter_rows, 
				int number_of_filter_cols,
				int number_of_filters, 	
				int number_of_images,		 
				int number_of_image_rows, 
				int number_of_image_cols,
				int depth, 
				__global int * restrict conv_pad_begin,
				__global int * restrict conv_pad_end,				
				int stride,
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	int num_rows_after_padding,num_cols_after_padding;
	num_rows_after_padding = number_of_image_rows + conv_pad_begin[0] + conv_pad_end[0];
	num_cols_after_padding = number_of_image_cols + conv_pad_begin[1] + conv_pad_end[1];
	__local double temp_image_padded[876096];
	index = 0;
	image_size = number_of_image_rows * number_of_image_cols;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <number_of_filters; layer++) 
		{
			//image_index = number_of_image_rows * number_of_image_cols * image_number;
			for(int d=0;d<depth;d++)
			{
				for (i = 0; i < number_of_image_rows; i+=stride) 
				{
					for (j = 0; j < number_of_image_cols; j+=stride) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - conv_pad_begin[0];
						int paderY = j - conv_pad_begin[1];

						 for(int filterX=0; filterX<number_of_filter_rows; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<number_of_filter_cols; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=number_of_image_rows||paderY<0||paderY>=number_of_image_cols)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*number_of_image_rows*number_of_image_cols*depth)+(d*number_of_image_rows*number_of_image_cols)+(number_of_image_cols*PaddedX)+PaddedY] *weights[(layer*number_of_filter_rows*number_of_filter_cols*depth)+(d*number_of_filter_rows*number_of_filter_cols)+(filterX*number_of_filter_cols)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - conv_pad_begin[1];
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}

