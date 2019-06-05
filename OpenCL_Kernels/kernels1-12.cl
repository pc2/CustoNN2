//combined
//conv1a

__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 224 * 224;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <64; layer++) 
		{
			
			for(int d=0;d<3;d++)
			{
				for (i = 0; i < 224; i+=2) 
				{
					for (j = 0; j < 224; j+=2) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 2;
						int paderY = j - 2;

						 for(int filterX=0; filterX<7; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<7; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=224||paderY<0||paderY>=224)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*224*224*3)+(d*224*224)+(224*PaddedX)+PaddedY] *weights[(layer*7*7*3)+(d*7*7)+(filterX*7)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 2;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


// conv2b

__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 56*56;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <64; layer++) 
		{
			
			for(int d=0;d<64;d++)
			{
				for (i = 0; i < 56; i+=2) 
				{
					for (j = 0; j < 56; j+=2) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 0;
						int paderY = j - 0;

						 for(int filterX=0; filterX<1; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<1; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=56||paderY<0||paderY>=56)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*56*56*64)+(d*56*56)+(56*PaddedX)+PaddedY] *weights[(layer*1*1*64)+(d*1*1)+(filterX*1)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 0;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv2c

__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 56*56;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <192; layer++) 
		{
			
			for(int d=0;d<64;d++)
			{
				for (i = 0; i < 56; i+=1) 
				{
					for (j = 0; j < 56; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<3; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=56||paderY<0||paderY>=56)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*56*56*64)+(d*56*56)+(56*PaddedX)+PaddedY] *weights[(layer*3*3*64)+(d*3*3)+(filterX*3)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 1;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv3b1
__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28*28;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <64; layer++) 
		{
			
			for(int d=0;d<192;d++)
			{
				for (i = 0; i < 28; i+=1) 
				{
					for (j = 0; j < 28; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 0;
						int paderY = j - 0;

						 for(int filterX=0; filterX<1; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<1; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=28||paderY<0||paderY>=28)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*28*28*192)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*1*1*192)+(d*1*1)+(filterX*1)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 0;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv3b2
__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28*28;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <96; layer++) 
		{
			
			for(int d=0;d<192;d++)
			{
				for (i = 0; i < 28; i+=1) 
				{
					for (j = 0; j < 28; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 0;
						int paderY = j - 0;

						 for(int filterX=0; filterX<1; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<1; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=28||paderY<0||paderY>=28)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*28*28*192)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*1*1*192)+(d*1*1)+(filterX*1)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 0;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv3b3
__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28*28;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <128; layer++) 
		{
			
			for(int d=0;d<96;d++)
			{
				for (i = 0; i < 28; i+=1) 
				{
					for (j = 0; j < 28; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<13; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<3; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=28||paderY<0||paderY>=28)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*28*28*96)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*96)+(d*3*3)+(filterX*3)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 1;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv3b4

__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28*28;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <16; layer++) 
		{
			
			for(int d=0;d<192;d++)
			{
				for (i = 0; i < 28; i+=1) 
				{
					for (j = 0; j < 28; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 0;
						int paderY = j - 0;

						 for(int filterX=0; filterX<1; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<1; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=28||paderY<0||paderY>=28)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*28*28*192)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*1*1*192)+(d*1*1)+(filterX*1)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 0;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv3b5

__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28*28;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <32; layer++) 
		{
			
			for(int d=0;d<16;d++)
			{
				for (i = 0; i < 28; i+=1) 
				{
					for (j = 0; j < 28; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<3; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=28||paderY<0||paderY>=28)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*28*28*16)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*16)+(d*3*3)+(filterX*3)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 1;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv3b6

__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28*28;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <32; layer++) 
		{
			
			for(int d=0;d<192;d++)
			{
				for (i = 0; i < 28; i+=1) 
				{
					for (j = 0; j < 28; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 0;
						int paderY = j - 0;

						 for(int filterX=0; filterX<1; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<1; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=28||paderY<0||paderY>=28)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*28*28*192)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*1*1*192)+(d*1*1)+(filterX*1)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 0;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv3c1
__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28*28;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <128; layer++) 
		{
			
			for(int d=0;d<256;d++)
			{
				for (i = 0; i < 28; i+=1) 
				{
					for (j = 0; j < 28; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 0;
						int paderY = j - 0;

						 for(int filterX=0; filterX<1; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<1; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=28||paderY<0||paderY>=28)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*28*28*256)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*1*1*256)+(d*1*1)+(filterX*1)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 0;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv3c2
__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28*28;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <128; layer++) 
		{
			
			for(int d=0;d<256;d++)
			{
				for (i = 0; i < 28; i+=1) 
				{
					for (j = 0; j < 28; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 0;
						int paderY = j - 0;

						 for(int filterX=0; filterX<1; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<1; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=28||paderY<0||paderY>=28)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*28*28*256)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*1*1*256)+(d*1*1)+(filterX*1)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 0;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}


//conv3c3
__kernel void ConvolutionLayer(__global double * restrict img, 
				__global float * restrict weights, 
				__global float * restrict bias,
				 int number_of_images,		 
				__global double * restrict output){
	

	//printf("Inside conv layer\n");
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28*28;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <192; layer++) 
		{
			
			for(int d=0;d<128;d++)
			{
				for (i = 0; i < 28; i+=1) 
				{
					for (j = 0; j < 28; j+=1) 
					{
				
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++)
			      			 {
			    	  		 	for(int filterY=0; filterY<3; filterY++)
			    	  		 	{
								if(paderX<0||paderX>=28||paderY<0||paderY>=28)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*28*28*128)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*128)+(d*3*3)+(filterX*3)+filterY] ;
							//printf("%f = %f * %f\n",temp_conv_val,temp_image_padded[(PaddedX*num_cols_after_padding) + PaddedY],weights[(filterX*number_of_filter_cols)+filterY]);
			                			PaddedY++;
								}
								paderY++;
			           			 }
			           			PaddedX++;
							paderX++;
			           			PaddedY=j;
							paderY = j - 1;
						}

					
						output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
						index++;
				


				

					}
		
				}
			}
	

		}
	}
}



