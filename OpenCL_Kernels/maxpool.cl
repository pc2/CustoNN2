__kernel void MaxPool(__global int * restrict input, __global int * restrict img,int number_of_image_rows, int  number_of_image_cols,int number_of_filters, int stride,int padding,int number_of_images,__global int * restrict conv_pad_begin,__global int * restrict conv_pad_end,__global int * restrict output)
{

	int count = 0,modVal = number_of_image_cols - stride; //mod val is used to skip certain loops
	double maxpool[9]; //maxpooing matrix 1D matrix with 0 and 1 position has r1c1 r1c2 and 2 and 3 has r2c1 r2c2
	int num_rows_after_padding, num_cols_after_padding;
	num_rows_after_padding = number_of_image_rows + conv_pad_begin[0] + conv_pad_end[0];
	num_cols_after_padding = number_of_image_cols + conv_pad_begin[1] + conv_pad_end[1];
	int image_size = num_rows_after_padding * num_cols_after_padding;
	int image_number;
	int max;
	 
	
	
	__local double temp_image_padded[200704];

	for(int image_number=0;image_number<number_of_images;image_number++)
	{
		for(int f=0;f<number_of_filters;f++)
		{
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
                        			temp_image_padded[(num_cols_after_padding*r)+c]=img[(image_number*number_of_image_rows*number_of_image_cols*number_of_filters)+(f*number_of_image_rows*number_of_image_cols)+(number_of_image_cols*r_in)+c_in];
                        c_in++;
                    			}

                		}
                		if(r>(conv_pad_begin[0]-1)&&r<(number_of_image_rows+1))
                		{
                    			r_in++;
                		}
            		}
            //printf("Padding ends")

            		for(int i = 0; i<image_size;i+=stride)
			{
		
	
				maxpool[0] = temp_image_padded[i];
				maxpool[1] = temp_image_padded[i+1];
				maxpool[2] = temp_image_padded[i+2];
				maxpool[3] = temp_image_padded[i+number_of_image_cols];
				maxpool[4] = temp_image_padded[i+number_of_image_cols + 1];
				maxpool[5] = temp_image_padded[i+number_of_image_cols + 2];
				maxpool[6] = temp_image_padded[i+2*number_of_image_cols];
				maxpool[7] = temp_image_padded[i+2*number_of_image_cols+1];
				maxpool[8] = temp_image_padded[i+2*number_of_image_cols+2];


				if(i%modVal == 0 && i!=0)
				{
					modVal+=number_of_image_cols * stride;
					i+=number_of_image_cols* stride; //2 rows are considered at a time
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

		}
	}
}
