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
/*
 * Kernel for Avgpooling Layer
 * Used for reducing the dimension of images.
 */

/*
 * Kernel for Avgpooling Layer
 * Used for reducing the dimension of images.
 */

__kernel void AvgPool(__global double * restrict input, 
			int number_of_image_rows, 
			int  number_of_image_cols,
			int number_of_filters,
			int kernel_size, 
			int stride,
			int number_of_images,
			__global double * restrict output){
	double avgpool[200]; 
	int image_size = number_of_image_rows * number_of_image_cols * number_of_filters * number_of_images;
	int avg=0, i, oindex=0,count=kernel_size, s=kernel_size, k, f, startIndex=0, endIndex=s, imageIndex=1, j=0;

   	for(f=0;f<s;f++){
	   while(count!=0){
		for(k=startIndex;k<endIndex;k++){
		    avgpool[j]=input[k];
		    j++;  
		}
		count--; 
		startIndex=number_of_image_cols*imageIndex;
		imageIndex++;       
		endIndex=startIndex+s;       
	    }
   	}
	for(i=0;i<s*s;i++){
       		avg=avg+avgpool[i];
   	}
	avg=avg/(s*s);
	output[oindex]=avg;
	oindex++;
}




__kernel void ConcatLayer(__global double * restrict input_1, 
			  __global double * restrict input_2, 
			  __global double * restrict input_3, 
			  __global double * restrict input_4, 
			  int input_1_rows, int input_1_cols, 
			  int input_1_filters, 
			  int input_2_rows, int input_2_cols, 
			  int input_2_filters, 
			  int input_3_rows, int input_3_cols, 
			  int input_3_filters, 
			  int input_4_rows, int input_4_cols, 
			  int input_4_filters, 
			  __global double * restrict output){
	int image_layers = 1; //Since image is RGB we have 3 layers for each image
	int total_inputs = 4;
	//input_1  convolution layer input 10*10*3*4 where 4 is the number pf filters
	
	int temp_list[4][4]; 
	//to store dimensions [][0] 28x28 and [][1] filter number
	temp_list[0][0] = input_1_rows;
	temp_list[1][0] = input_2_rows;
	temp_list[2][0] = input_3_rows;
	temp_list[3][0] = input_4_rows;
	temp_list[0][1] = input_1_cols;
	temp_list[1][1] = input_2_cols;
	temp_list[2][1] = input_3_cols;
	temp_list[3][1] = input_4_cols;
	temp_list[0][2] = input_1_filters;
	temp_list[1][2] = input_2_filters;
	temp_list[2][2] = input_3_filters;
	temp_list[3][2] = input_4_filters;
	

	for(int i = 1; i<=total_inputs; i++){
		double temp_var[100000];
		if(i==1){
			for(int x = 1; x<=temp_list[(i-1)][0]; x++){
				for(int y = 1; y<=temp_list[(i-1)][1]; y++){
					temp_var[x] = input_1[(x*y)-1];
				}
			}
		}
		if(i==2){
			for(int x = 1; x<=temp_list[(i-1)][0]; x++){
				for(int y = 1; y<=temp_list[(i-1)][1]; y++){
					temp_var[x] = input_2[(x*y)-1];
				}
			}
		}
		if(i==3){
			for(int x = 1; x<=temp_list[(i-1)][0]; x++){
				for(int y = 1; y<=temp_list[(i-1)][1]; y++){
					temp_var[x] = input_3[(x*y)-1];
				}
			}
		}
		if(i==4){
			for(int x = 1; x<=temp_list[(i-1)][0]; x++){
				for(int y = 1; y<=temp_list[(i-1)][1]; y++){
					temp_var[x] = input_4[(x*y)-1];
				}
			}
		}
		for(int filter=1; filter<=temp_list[(i-1)][2]; filter++){
			for(int layer=1; layer <= image_layers; layer++){		
				for(int c=1; c <= temp_list[(i-1)][1]; c++){
					for(int r=1; r <= temp_list[i][0]; r++){
					
						output[(i*r*c*layer*filter)-1]+=temp_var[(r*c*layer*filter)-1];

					}
				}
			}
		}

	}
	


}
