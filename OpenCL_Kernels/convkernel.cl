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
		//printf("Before padding\n");
		//padding logic
		for(int d=0;d<depth;d++)
		{
			for(int r=0,r_in=0;r<num_rows_after_padding && r_in<number_of_image_rows;r++)
			{
				for(int c=0,c_in=0;c<num_cols_after_padding && c_in<number_of_image_cols;c++)
				{
					if((r<conv_pad_begin[0]||r>(number_of_image_rows+1))||(c<conv_pad_begin[1]||c>(number_of_image_cols+1)))
					{
						temp_image_padded[(d*num_rows_after_padding*num_cols_after_padding)+(num_cols_after_padding*r)+c]=0.0;
					}
					else
					{
						temp_image_padded[(d*num_rows_after_padding*num_cols_after_padding)+(num_cols_after_padding*r)+c]=img[(image_number*number_of_image_rows*number_of_image_cols*depth)+(d*number_of_image_rows*number_of_image_cols)+(number_of_image_cols*r_in)+c_in];
						c_in++;
					}

				}
				if(r>(conv_pad_begin[0]-1)&&r<(number_of_image_rows+1))
				{
					r_in++;
				}
			}
		}
		//printf("Padding done\n");
		
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

						 for(int filterX=0; filterX<number_of_filter_rows; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<number_of_filter_cols; filterY++)
			    	  		 	{
			               			 	temp_conv_val  += temp_image_padded[(d*num_rows_after_padding*num_cols_after_padding)+(PaddedX*num_cols_after_padding) + PaddedY] *weights[(layer*number_of_filter_rows*number_of_filter_cols*depth)+(d*number_of_filter_rows*number_of_filter_cols)+(filterX*number_of_filter_cols)+filterY] ;
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
}

