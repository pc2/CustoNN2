//conv5b4

__kernel void InceptionV1_InceptionV1_Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,__global float* restrict input2) {
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
        }
        compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)])>0?compute[((((ff * 7) + yy) * 7) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv5b5

__kernel void InceptionV1_InceptionV1_Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D(__global double * restrict img, 
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


//conv5b6
__kernel void InceptionV1_InceptionV1_Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,__global float* restrict input2 ) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff] ;
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
        }
        compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)])>0?compute[((((ff * 7) + yy) * 7) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv5c1
__kernel void InceptionV1_InceptionV1_Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 384; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
        }
        compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)])>0? (compute[((((ff * 7) + yy) * 7) + xx)]):0.000000e+00f;
      }
    }
  }
}

//conv5c2

__kernel void InceptionV1_InceptionV1_Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,__global float* restrict input2) {
  for (int ff = 0; ff < 192; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
        }
        compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)])>0?compute[((((ff * 7) + yy) * 7) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv5c3

__kernel void InceptionV1_InceptionV1_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
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


//conv5c4
__kernel void InceptionV1_InceptionV1_Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 48; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
        }
        compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)])>0?compute[((((ff * 7) + yy) * 7) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv5c5

__kernel void InceptionV1_InceptionV1_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
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

//conv5c6

__kernel void InceptionV1_InceptionV1_Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,  __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
        }
        compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)])>0?compute[((((ff * 7) + yy) * 7) + xx)]:0.000000e+00f;
      }
    }
  }
}

//conv0c

__kernel void InceptionV1_Logits_Conv2d_0c_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 1001; ++ff) {
    compute[ff] = 0.000000e+00f;
    for (int rc = 0; rc < 1024; ++rc) {
      compute[ff] = (compute[ff] + (input0[rc] * input1[((ff * 1024) + rc)]));
    }
  }
}

