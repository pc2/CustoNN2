__kernel void MaxPool(__global int * restrict input, 
            int number_of_image_rows, 
            int number_of_image_cols, 
            int number_of_filters, 
            int stride, 
            int number_of_images, 
            __global int * restrict conv_pad_begin,
            __global int * restrict conv_pad_end,
            __global int * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	int num_rows_after_padding,num_cols_after_padding;
	num_rows_after_padding = number_of_image_rows + conv_pad_begin[0] + conv_pad_end[0];
	num_cols_after_padding = number_of_image_cols + conv_pad_begin[1] + conv_pad_end[1];
	//__local double temp_image_padded[876096];
	index = 0;
	image_size = number_of_image_rows * number_of_image_cols;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <number_of_filters; layer++) 
		{
			//image_index = number_of_image_rows * number_of_image_cols * image_number;
			
			for (i = 0; i < number_of_image_rows; i+=stride) 
			{
				for (j = 0; j < number_of_image_cols; j+=stride) 
				{
				
					//double temp_conv_val = bias[layer];
					int PaddedX = i;
					int PaddedY = j;
					int paderX = i - conv_pad_begin[0];
					int paderY = j - conv_pad_begin[1];
					double maxpool[9];
					int pool_index = 0;
					for(int filterX=0; filterX<3; filterX++)
			      		{
			    	  		 for(int filterY=0; filterY<3; filterY++)
			    	  		 {
							if(paderX<0||paderX>=number_of_image_rows||paderY<0||paderY>=number_of_image_cols)
							{
								maxpool[pool_index] = 0;
								pool_index++;
							}
							else
							{
			               			 	maxpool[pool_index]  = input[(image_number*number_of_image_rows*number_of_image_cols*number_of_filters)+(layer*number_of_image_rows*number_of_image_cols)+(number_of_image_cols*PaddedX)+PaddedY];
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								pool_index++;
							}
							paderY++;
			           		}
			           		PaddedX++;
						paderX++;
			           		PaddedY=j;
						paderY = j - conv_pad_begin[1];
					}

					double max = 0;
					for(int m=0;m<9;m++)
					{
						if(max<maxpool[m])
							max = maxpool[m];
					}
					output[index] = max;
					index++;
				


				

				}
		
			}
			
	

		}
	}
}
