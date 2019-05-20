#pragma OPENCL EXTENSION cl_intel_channels : enable


typedef struct conv_buffer {
        double temp_buffer;
}co;

//To send 256 bits of data
typedef struct max_buffer {
        double maxPool_buffer[4];
}maxStruct;

channel double ConvOutChannel __attribute__((depth(64)));                        
//Channel Between Maxpool and FC Layer
channel maxStruct MaxPoolOutChannel __attribute__((depth(28))) __attribute__((io("kernel_input_ch0"))); // Channel Tx
channel maxStruct FCInChannel __attribute__((depth(28))) __attribute__((io("kernel_input_ch0")));  // Channel Rx



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
				int stride){
	

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
		 
    	//printf("Convolution for Image %d\n",image_number);


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
									
			                		PaddedY++;
			           		 }
			           		PaddedX++;
			           		PaddedY=j;
					}

					
					double co1 = (temp_conv_val>0) ? temp_conv_val : 0;
					//if(image_number==0 && layer==4)
                    //    printf("%f  ",co1);

					write_channel_intel(ConvOutChannel,co1);
					index++;
				}
				//if(image_number==0 && layer==4)
                //	printf("\n");

		
			}
			//if(image_number==0 && layer==4)
              //  	printf("\n\n");

		}
	}
}


/*
 * Kernel for Maxpooling Layer
 * Used for reducing the dimension of images. We use a 2*2 matrix for maxpooling.
 */

__kernel void MaxPool(
			int number_of_image_rows, 
			int  number_of_image_cols,
			int number_of_filters, 
			int stride,
			int number_of_images){
	int count = 0;
	double maxpool[4]; //maxpooing matrix 1D matrix with 0 and 1 position has r1c1 r1c2 and 2 and 3 has r2c1 r2c2
	for ( int i =0; i < number_of_images; ++i)
        {
			printf("Maxpool for Image %d\n",i);
            
            for (int k = 0; k <number_of_filters; ++k)
            {
				double img[28*28];
                        //Store the Channels data 5of 1 Image in a linear array.
                        
                for ( int j = 0; j<28; j++ ) {
                    //struct conv_buffer conv1 = read_channel_intel(ConvOutChannel);
                    #pragma unroll
                    for(int l=0; l<28; l++) {
                        img[(j*28)+l]=read_channel_intel(ConvOutChannel);
                    }
                }


                for (int x = 0; x < number_of_image_rows; x=x+stride)
                {
					struct max_buffer max1;
					int pixelCount=0;
                    for (int y = 0; y < number_of_image_cols; y=y+stride)
                    {
						double max=0.0;
                        maxpool[0] = img[(x*number_of_image_cols)+(y)];
                        maxpool[1] = img[(x*number_of_image_cols)+(y+1)];
                        maxpool[2] = img[(x*number_of_image_cols)+(y+number_of_image_cols)];
                        maxpool[3] = img[(x*number_of_image_cols)+(y+number_of_image_cols+1)];
                                        
						max = maxpool[0];
                        for(int j = 1; j<4; j++)
						{		
        					if(maxpool[j] > max)
        						max = maxpool[j];
						}
						max1.maxPool_buffer[pixelCount]=max;
						pixelCount++;
						//Check if 256 bits has been added to the channel
						if(pixelCount==4){
							pixelCount=0;
							write_channel_intel(MaxPoolOutChannel,max1);
						}
						//output[count] = max;
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

__kernel void FCL_Kernel(__global float * restrict weights,
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
		printf("FC for Image %d\n",image_number);
		double maxScore=0.0;
		int weightIndex=0;
		double maxpooldata[14*14*32];

		//Read 256 bits input and store it in a local array
		int imgIndex=0;
		for(int i=0; i<32; i++) {
			for(int q=0; q<7; q++) {
                struct max_buffer max1[7];
                for(int colData=0;colData<7;colData++){                                
                    max1[colData]= read_channel_intel(FCInChannel);
                }
                               
            	for(int colData=0;colData<7;colData++){
                    for(int l=0;l<4;l++){
                        maxpooldata[imgIndex] = max1[colData].maxPool_buffer[l];
						imgIndex++;
						printf(" Iamge Index: %d",imgIndex);
                    }
                }
            }
		}
		

		int curr_pos = image_number * image_size;
		//double  sum=0.0;
		for(int w = 0; w < weight_number; w++){
			double sum=bias[w];
			for(int j=0;j<image_size;j++){
				sum+= maxpooldata[j] * weights[(w*image_size)+j];
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
