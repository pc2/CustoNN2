__kernel void Conv2d_1a_7x7_Conv2D(__global unsigned char * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global float * restrict output){

  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 224 * 224;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <64; layer++){
	 float temp_conv_val = bias[layer];
    
        for (i = 0; i < 224; i+=2)
	{
          for (j = 0; j < 224; j+=2)
	{
        	 for(int d=0;d<3;d++)
		{
           
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 2;
            int paderY = j - 2;

             for(int filterX=0; filterX<7; filterX++)
		{
                    for(int filterY=0; filterY<7; filterY++)
			{
                if(paderX<0||paderX>=224||paderY<0||paderY>=224){}
		else{
                            temp_conv_val  += img[(image_number*224*224*3)+(d*224*224)+(224*PaddedX)+PaddedY] *weights[(layer*7*7*3)+(d*7*7)+(filterX*7)+filterY] ;
                            PaddedY++;
                }
                paderY++;
             }
              PaddedX++;
              paderX++;
              PaddedY=j;
              paderY = j - 2;
            }
		}
            output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
            index++;
          }
        }
      
    }
  }
}


__kernel void MaxPool_2a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0){
  for (int ax1 = 0; ax1 < 64; ++ax1) {
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = max(tensor[((((ax1 * 56) + ax2) * 56) + ax3)], (float)((((ax2 * 2) < (112 - rv)) && ((ax3 * 2) < (112 - rv1))) ? input0[((((((((ax1 * 56) + ax2) * 2) + rv) * 56) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void Conv2d_2b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
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


__kernel void Conv2d_2c_3x3_Conv2D(__global float * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global float * restrict output){
  
  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 56*56;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <192; layer++) 
	{
	float temp_conv_val = bias[layer];
      
        for (i = 0; i < 56; i+=1) {
          for (j = 0; j < 56; j+=1) {
            	for(int d=0;d<64;d++){
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 1;
            int paderY = j - 1;

             for(int filterX=0; filterX<3; filterX++){
                for(int filterY=0; filterY<3; filterY++){
                  if(paderX<0||paderX>=56||paderY<0||paderY>=56){}else{
                      temp_conv_val  += img[(image_number*56*56*64)+(d*56*56)+(56*PaddedX)+PaddedY] *weights[(layer*3*3*64)+(d*3*3)+(filterX*3)+filterY] ;
                      PaddedY++;
                  }
                paderY++;
                }
                PaddedX++;
                paderX++;
                PaddedY=j;
                paderY = j - 1;
            }
		}
            output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
            index++;
          }
        }
      
    }
  }
}

__kernel void MaxPool_3a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 192; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((ax2 * 2) < (56 - rv)) && ((ax3 * 2) < (56 - rv1))) ? input0[((((((((ax1 * 28) + ax2) * 2) + rv) * 28) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}
