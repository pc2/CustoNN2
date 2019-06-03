__kernel void InceptionV1_InceptionV1_Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1
									__global float* restrict input2	){
	

for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
        }
	compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)]>0) ? + compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;

      }
    }
  }
}




__kernel void InceptionV1_InceptionV1_Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
									__global float * restrict weights, 
									__global float * restrict bias,
							 		int number_of_images,		 
									__global double * restrict output) {

int i,j,k;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28 * 28;
	for (image_number = 0; image_number < number_of_images; image_number++){
		for (layer = 0; layer <96; layer++){
			for(int d=0;d<32;d++){
				for (i = 0; i < 28; i+=1){
					for (j = 0; j < 28; j+=1){
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++){
			    	  		 	for(int filterY=0; filterY<3; filterY++){
									if(paderX<0||paderX>=28||paderY<0||paderY>=28)
									{
									}
								else{
			               			 	temp_conv_val  += img[(image_number*28*28*32)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*32)+(d*3*3)+		(filterX*3)+filterY] ;
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


__kernel void InceptionV1_InceptionV1_Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1
									__global float* restrict input2	){
	

for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
        }
	compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)]>0) ? + compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;

      }
    }
  }
}


__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1
									__global float* restrict input2	){
	

for (int ff = 0; ff < 192; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
        }
	compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? + compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;

      }
    }
  }
}



__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1
									__global float* restrict input2	){
	

for (int ff = 0; ff < 96; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
        }
	compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? + compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;

      }
    }
  }
}



__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
									__global float * restrict weights, 
									__global float * restrict bias,
							 		int number_of_images,		 
									__global double * restrict output) {

int i,j,k;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 28 * 28;
	for (image_number = 0; image_number < number_of_images; image_number++){
		for (layer = 0; layer <208; layer++){
			for(int d=0;d<96;d++){
				for (i = 0; i < 14; i+=1){
					for (j = 0; j < 14; j+=1){
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++){
			    	  		 	for(int filterY=0; filterY<3; filterY++){
									if(paderX<0||paderX>=14||paderY<0||paderY>=14)
									{
									}
								else{
			               			 	temp_conv_val  += img[(image_number*14*14*96)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*96)+(d*3*3)+		(filterX*3)+filterY] ;
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

__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1
									__global float* restrict input2	){
	

for (int ff = 0; ff < 16; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
        }
	compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? + compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;

      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
									__global float * restrict weights, 
									__global float * restrict bias,
							 		int number_of_images,		 
									__global double * restrict output) {

int i,j,k;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 14 * 14;
	for (image_number = 0; image_number < number_of_images; image_number++){
		for (layer = 0; layer <48; layer++){
			for(int d=0;d<16;d++){
				for (i = 0; i < 14; i+=1){
					for (j = 0; j < 14; j+=1){
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++){
			    	  		 	for(int filterY=0; filterY<3; filterY++){
									if(paderX<0||paderX>=14||paderY<0||paderY>=14)
									{
									}
								else{
			               			 	temp_conv_val  += img[(image_number*14*14*16)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*16)+(d*3*3)+		(filterX*3)+filterY] ;
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

__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1
									__global float* restrict input2	){
	

for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
        }
	compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? + compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;

      }
    }
  }
}


__kernel void InceptionV1_InceptionV1_Mixed_4c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1
									__global float* restrict input2	){
	

for (int ff = 0; ff < 160; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
        }
	compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? + compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;

      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4c_Branch_1_Conv2d_0a_1x1_Conv2D__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1
									__global float* restrict input2	){
	

for (int ff = 0; ff < 112; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
	compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? + compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;

      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4c_Branch_1_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
									__global float * restrict weights, 
									__global float * restrict bias,
							 		int number_of_images,		 
									__global double * restrict output) {

int i,j,k;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 14 * 14;
	for (image_number = 0; image_number < number_of_images; image_number++){
		for (layer = 0; layer <224; layer++){
			for(int d=0;d<112;d++){
				for (i = 0; i < 14; i+=1){
					for (j = 0; j < 14; j+=1){
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++){
			    	  		 	for(int filterY=0; filterY<3; filterY++){
									if(paderX<0||paderX>=14||paderY<0||paderY>=14)
									{
									}
								else{
			               			 	temp_conv_val  += img[(image_number*14*14*112)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*112)+(d*3*3)+	(filterX*3)+filterY] ;
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

