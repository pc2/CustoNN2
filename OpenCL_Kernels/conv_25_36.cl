__kernel void Mixed_4c_Branch_2_Conv2d_0a_1x1(__global float* restrict compute, 
							__global float* restrict input0, 
							__global float* restrict input1, 
							__global float* restrict input2) {
  for (int ff = 0; ff < 24; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}

__kernel void Mixed_4c_Branch_2_Conv2d_0b_3x3(__global double * restrict img, 
							__global float * restrict weights, 
							__global float * restrict bias,
							 int number_of_images,		 
							__global double * restrict output){
	int i,j,k;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 14 * 14;
	for (image_number = 0; image_number < number_of_images; image_number++){
		for (layer = 0; layer <64; layer++){
			for(int d=0;d<24;d++){
				for (i = 0; i < 14; i+=1){
					for (j = 0; j < 14; j+=1){
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++){
			    	  		 	for(int filterY=0; filterY<3; filterY++){
									if(paderX<0||paderX>=14||paderY<0||paderY>=14){}else{
			               			 	temp_conv_val  += img[(image_number*14*14*24)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*24)+(d*3*3)+(filterX*3)+filterY] ;
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

__kernel void Mixed_4c_Branch_3_Conv2d_0b_1x1(__global float* restrict compute, 
							__global float* restrict input0, 
							__global float* restrict input1, 
							__global float* restrict input2) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}


__kernel void Mixed_4d_Branch_0_Conv2d_0a_1x1(__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1, 
									__global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}

__kernel void Mixed_4d_Branch_1_Conv2d_0a_1x1(__global float* restrict compute, 
									__global float* restrict input0, 
									__global float* restrict input1, 
									__global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}

__kernel void Mixed_4d_Branch_1_Conv2d_0b_3x3(__global double * restrict img, 
							__global float * restrict weights, 
							__global float * restrict bias,
							 int number_of_images,		 
							__global double * restrict output){
	int i,j,k;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 14 * 14;
	for (image_number = 0; image_number < number_of_images; image_number++){
		for (layer = 0; layer <256; layer++){
			for(int d=0;d<128;d++){
				for (i = 0; i < 14; i+=1){
					for (j = 0; j < 14; j+=1){
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++){
			    	  		 	for(int filterY=0; filterY<3; filterY++){
									if(paderX<0||paderX>=14||paderY<0||paderY>=14){}else{
			               			 	temp_conv_val  += img[(image_number*14*14*128)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*128)+(d*3*3)+(filterX*3)+filterY] ;
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

__kernel void Mixed_4d_Branch_2_Conv2d_0a_1x1(__global float* restrict compute, 
											__global float* restrict input0, 
											__global float* restrict input1, 
											__global float* restrict input2) {
  for (int ff = 0; ff < 24; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}


__kernel void Mixed_4c_Branch_2_Conv2d_0b_3x3(__global double * restrict img, 
											__global float * restrict weights, 
											__global float * restrict bias,
											 int number_of_images,		 
											__global double * restrict output){
	int i,j,k;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 14 * 14;
	for (image_number = 0; image_number < number_of_images; image_number++){
		for (layer = 0; layer <64; layer++){
			for(int d=0;d<24;d++){
				for (i = 0; i < 14; i+=1){
					for (j = 0; j < 14; j+=1){
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++){
			    	  		 	for(int filterY=0; filterY<3; filterY++){
									if(paderX<0||paderX>=14||paderY<0||paderY>=14){}else{
			               			 	temp_conv_val  += img[(image_number*14*14*24)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*24)+(d*3*3)+(filterX*3)+filterY] ;
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

__kernel void Mixed_4d_Branch_3_Conv2d_0b_1x1(__global float* restrict compute, 
											__global float* restrict input0, 
											__global float* restrict input1, 
											__global float* restrict input2) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}

__kernel void Mixed_4e_Branch_0_Conv2d_0a_1x1(__global float* restrict compute, 
											__global float* restrict input0, 
											__global float* restrict input1, 
											__global float* restrict input2) {
  for (int ff = 0; ff < 112; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}

__kernel void Mixed_4e_Branch_1_Conv2d_0a_1x1(__global float* restrict compute, 
											__global float* restrict input0, 
											__global float* restrict input1, 
											__global float* restrict input2) {
  for (int ff = 0; ff < 144; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}

__kernel void Mixed_4e_Branch_1_Conv2d_0b_3x3(__global double * restrict img, 
											__global float * restrict weights, 
											__global float * restrict bias,
											 int number_of_images,		 
											__global double * restrict output){
	int i,j,k;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 14 * 14;
	for (image_number = 0; image_number < number_of_images; image_number++){
		for (layer = 0; layer <288; layer++){
			for(int d=0;d<144;d++){
				for (i = 0; i < 14; i+=1){
					for (j = 0; j < 14; j+=1){
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++){
			    	  		 	for(int filterY=0; filterY<3; filterY++){
									if(paderX<0||paderX>=14||paderY<0||paderY>=14){}else{
			               			 	temp_conv_val  += img[(image_number*14*14*144)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*144)+(d*3*3)+(filterX*3)+filterY] ;
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


