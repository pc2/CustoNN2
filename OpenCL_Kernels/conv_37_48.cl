//37
__kernel void Mixed_4e_Branch_2_Conv2d_0a_1x1(__global float* restrict compute, 
					      __global float* restrict input0, 
					      __global float* restrict input1, 
					      __global float* restrict input2) {
  for (int ff = 0; ff < 32; ++ff) {
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



//38
__kernel void Mixed_4e_Branch_2_Conv2d_0b_3x3(__global double * restrict img, 
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
			for(int d=0;d<32;d++){
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
			               			 	temp_conv_val  += img[(image_number*14*14*32)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*32)+(d*3*3)+(filterX*3)+filterY] ;
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


//39

__kernel void Mixed_4e_Branch_3_Conv2d_0b_1x1(__global float* restrict compute, 
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


//40

__kernel void Mixed_4f_Branch_0_Conv2d_0a_1x1(__global float* restrict compute, 
					      __global float* restrict input0, 
					      __global float* restrict input1, 
					      __global float* restrict input2) {
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 528; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 528) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}

//41
__kernel void Mixed_4f_Branch_1_Conv2d_0a_1x1(__global float* restrict compute, 
					      __global float* restrict input0, 
					      __global float* restrict input1, 
					      __global float* restrict input2) {
  for (int ff = 0; ff < 160; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 528; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 528) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}



//42

__kernel void Mixed_4f_Branch_1_Conv2d_0b_3x3(__global double * restrict img, 
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
		for (layer = 0; layer <320; layer++){
			for(int d=0;d<160;d++){
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
			               			 	temp_conv_val  += img[(image_number*14*14*320)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*320)+(d*3*3)+(filterX*3)+filterY] ;
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

//43
__kernel void Mixed_4f_Branch_2_Conv2d_0a_1x1(__global float* restrict compute, 
					      __global float* restrict input0, 
					      __global float* restrict input1, 
					      __global float* restrict input2) {
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 528; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 528) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}


//44
__kernel void Mixed_4f_Branch_2_Conv2d_0b_3x3(__global double * restrict img, 
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
		for (layer = 0; layer <128; layer++){
			for(int d=0;d<32;d++){
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
			               			 	temp_conv_val  += img[(image_number*14*14*32)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*32)+(d*3*3)+(filterX*3)+filterY] ;
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

//45

__kernel void Mixed_4f_Branch_3_Conv2d_0b_1x1(__global float* restrict compute, 
					      __global float* restrict input0, 
					      __global float* restrict input1, 
					      __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 528; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 528) + rc)]));
        }
        compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)]>0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}


//46
__kernel void Mixed_5b_Branch_0_Conv2d_0a_1x1(__global float* restrict compute, 
					      __global float* restrict input0, 
					      __global float* restrict input1, 
					      __global float* restrict input2) {
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
        }
        compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)]>0) ? compute[((((ff * 7) + yy) * 7) + xx)] : 0.000000e+00f;
      }
    }
  }
}

//47
__kernel void Mixed_5b_Branch_1_Conv2d_0a_1x1(__global float* restrict compute, 
					      __global float* restrict input0, 
					      __global float* restrict input1, 
					      __global float* restrict input2) {
  for (int ff = 0; ff < 160; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
        }
        compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)]>0) ? compute[((((ff * 7) + yy) * 7) + xx)] : 0.000000e+00f;
      }
    }
  }
}


//48
__kernel void Mixed_5b_Branch_1_Conv2d_0b_3x3(__global double * restrict img, 
											__global float * restrict weights, 
											__global float * restrict bias,
											 int number_of_images,		 
											__global double * restrict output){
	int i,j,k;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;
	int image_number;
	index = 0;
	image_size = 7 * 7;
	for (image_number = 0; image_number < number_of_images; image_number++){
		for (layer = 0; layer <320; layer++){
			for(int d=0;d<160;d++){
				for (i = 0; i < 7; i+=1){
					for (j = 0; j < 7; j+=1){
						double temp_conv_val = bias[layer];
						int PaddedX = i;
						int PaddedY = j;
						int paderX = i - 1;
						int paderY = j - 1;

						 for(int filterX=0; filterX<3; filterX++){
			    	  		 	for(int filterY=0; filterY<3; filterY++){
									if(paderX<0||paderX>=7||paderY<0||paderY>=7){}else{
			               			 	temp_conv_val  += img[(image_number*7*7*160)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*3*3*160)+(d*3*3)+(filterX*3)+filterY] ;
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