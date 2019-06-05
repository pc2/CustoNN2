//conv1a

__kernel void InceptionV1_InceptionV1_Conv2d_1a_7x7_Conv2D(__global double * restrict img, 
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

__kernel void InceptionV1_InceptionV1_Conv2d_2b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        compute[((((ff * 56) + yy) * 56) + xx)] = input2[ff] ;
        for (int rc = 0; rc < 64; ++rc) {
          compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 64) + rc)]));
        }
        compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)]>0)?compute[((((ff * 56) + yy) * 56) + xx)]:0.000000e+00f;
      }
    }
  }
}


//conv2c

__kernel void InceptionV1_InceptionV1_Conv2d_2c_3x3_Conv2D(__global double * restrict img, 
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
__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
        }
        compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)]>0)?compute[((((ff * 28) + yy) * 28) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv3b2
__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1 , __global float* restrict input2) {
  for (int ff = 0; ff < 96; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
        }
        compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)]>0)? compute[((((ff * 28) + yy) * 28) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv3b3
__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
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

__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 16; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff] ;
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
        }
        compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)]>0)?compute[((((ff * 28) + yy) * 28) + xx)]: 0.000000e+00f;
      }
    }
  }
}

//conv3b5

__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
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

__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1 , __global float* restrict input2) {
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff] ;
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
        }
        compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)]>0)? compute[((((ff * 28) + yy) * 28) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv3c1
__kernel void InceptionV1_InceptionV1_Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1 , __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
        }
        compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)]>0)? compute[((((ff * 28) + yy) * 28) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv3c2
__kernel void InceptionV1_InceptionV1_Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1 , __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
        }
        compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)]>0)? compute[((((ff * 28) + yy) * 28) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv3c3
__kernel void InceptionV1_InceptionV1_Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
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



