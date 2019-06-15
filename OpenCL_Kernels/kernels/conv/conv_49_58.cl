// Conv5b4
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
	image_size = 7*7;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <32; layer++) 
		{
			
			for(int d=0;d<832;d++)
			{
				for (i = 0; i < 7; i+=1) 
				{
					for (j = 0; j < 7; j+=1) 
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
								if(paderX<0||paderX>=7||paderY<0||paderY>=7)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*7*7*832)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*1*1*832)+(d*1*1)+(filterX*1)+filterY] ;
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
//Conv5b5
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
	image_size = 7*7;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <128; layer++) 
		{
			
			for(int d=0;d<32;d++)
			{
				for (i = 0; i < 7; i+=1) 
				{
					for (j = 0; j < 7; j+=1) 
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
								if(paderX<0||paderX>=7||paderY<0||paderY>=7)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*7*7*32)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*3*3*32)+(d*3*3)+(filterX*3)+filterY] ;
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

//Conv5b6
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
	image_size = 7*7;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <128; layer++) 
		{
			
			for(int d=0;d<832;d++)
			{
				for (i = 0; i < 7; i+=1) 
				{
					for (j = 0; j < 7; j+=1) 
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
								if(paderX<0||paderX>=7||paderY<0||paderY>=7)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*7*7*832)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*1*1*832)+(d*1*1)+(filterX*1)+filterY] ;
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



//Conv5c1
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
	image_size = 7*7;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <384; layer++) 
		{
			
			for(int d=0;d<832;d++)
			{
				for (i = 0; i < 7; i+=1) 
				{
					for (j = 0; j < 7; j+=1) 
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
								if(paderX<0||paderX>=7||paderY<0||paderY>=7)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*7*7*832)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*1*1*832)+(d*1*1)+(filterX*1)+filterY] ;
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




//Conv5c2
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
	image_size = 7*7;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <192; layer++) 
		{
			
			for(int d=0;d<832;d++)
			{
				for (i = 0; i < 7; i+=1) 
				{
					for (j = 0; j < 7; j+=1) 
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
								if(paderX<0||paderX>=7||paderY<0||paderY>=7)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*7*7*832)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*1*1*832)+(d*1*1)+(filterX*1)+filterY] ;
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



//Conv5c3
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
	image_size = 7*7;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <384; layer++) 
		{
			
			for(int d=0;d<192;d++)
			{
				for (i = 0; i < 7; i+=1) 
				{
					for (j = 0; j < 7; j+=1) 
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
								if(paderX<0||paderX>=7||paderY<0||paderY>=7)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*7*7*192)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*3*3*192)+(d*3*3)+(filterX*3)+filterY] ;
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



//Conv5c4
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
	image_size = 7*7;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <48; layer++) 
		{
			
			for(int d=0;d<832;d++)
			{
				for (i = 0; i < 7; i+=1) 
				{
					for (j = 0; j < 7; j+=1) 
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
								if(paderX<0||paderX>=7||paderY<0||paderY>=7)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*7*7*832)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*1*1*832)+(d*1*1)+(filterX*1)+filterY] ;
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



//Conv5c5
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
	image_size = 7*7;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <48; layer++) 
		{
			
			for(int d=0;d<128;d++)
			{
				for (i = 0; i < 7; i+=1) 
				{
					for (j = 0; j < 7; j+=1) 
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
								if(paderX<0||paderX>=7||paderY<0||paderY>=7)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*7*7*128)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*3*3*128)+(d*3*3)+(filterX*3)+filterY] ;
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



//Conv5c6
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
	image_size = 7*7;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <128; layer++) 
		{
			
			for(int d=0;d<832;d++)
			{
				for (i = 0; i < 7; i+=1) 
				{
					for (j = 0; j < 7; j+=1) 
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
								if(paderX<0||paderX>=7||paderY<0||paderY>=7)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*7*7*832)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*1*1*832)+(d*1*1)+(filterX*1)+filterY] ;
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



//Conv0c
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
	image_size = 1*1;
	for (image_number = 0; image_number < number_of_images; image_number++)
	{
		
		
		for (layer = 0; layer <1001; layer++) 
		{
			
			for(int d=0;d<1024;d++)
			{
				for (i = 0; i < 1; i+=1) 
				{
					for (j = 0; j < 1; j+=1) 
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
								if(paderX<0||paderX>=1||paderY<0||paderY>=1)
								{
									//do nothing here
								}
								else
								{
			               			 	temp_conv_val  += img[(image_number*1*1*1024)+(d*1*1)+(1*PaddedX)+PaddedY] *weights[(layer*1*1*1024)+(d*1*1)+(filterX*1)+filterY] ;
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





