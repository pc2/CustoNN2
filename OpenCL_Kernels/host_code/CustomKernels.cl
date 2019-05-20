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



__kernel void MaxPool(__global int * restrict img,
			int number_of_image_rows, 
			int  number_of_image_cols,
			int number_of_filters, 
			int stride,
			int number_of_images,
			__global int * restrict conv_pad_begin,
			__global int * restrict conv_pad_end,
			__global int * restrict output)
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
    int loopend=0;
    //int image_size = number_of_image_rows * number_of_image_cols * number_of_filters * number_of_images;
    int bigstart=0;
    int avg=0, rowi=1, i, oindex=0,count=3, s=9, k, f, startIndex=0, endIndex=3, bi=1, imageIndex=1, j=0, acII=3,inputarrayII=0,accincreement=0;
        int stridecount=1;
            for(int kk=0;kk<9;kk++){
                    imageIndex=1;
                    count=3;
                    loopend=startIndex+3;
                    while(count!=0){
                        for(k=startIndex;k<loopend;k++){
                                avgpool[j]=input[k];
                                j++; 
                        }
                        count--;
                        startIndex=loopend+6;
                        imageIndex++;  
                        loopend=startIndex+3;
                        endIndex=startIndex+s;  
                    }
                    for(i=0;i<j;i++)
                       avg=avg+avgpool[i];
                    avg=avg/s;
                    //cout<<"avg is "<<avg<<"\n";
                    output[oindex]=avg;
                    oindex++;
                    avg=0;
                    j=0;
                    if(stridecount%3==0){
                            bigstart=3*3*3*rowi;
                            startIndex=bigstart;
                            rowi++;
                    }else{
                        bigstart=bigstart+3;
                    }
                    startIndex=bigstart;
                    stridecount++;
                    accincreement++;
            }
    //for(i=0;i<9;i++)
        //cout<<output[i]<<"\t";
   

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
