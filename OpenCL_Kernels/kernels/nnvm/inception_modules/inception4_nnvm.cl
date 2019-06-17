__kernel void Mixed_4c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 160; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
      }
    }
  }
}

__kernel void Mixed_4c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
                  __global float* restrict input0, 
                  __global float* restrict input1,
                  __global float* restrict input2 ){
  

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

__kernel void Mixed_4c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
                  __global float * restrict weights, 
                  __global float * restrict bias,
                  int number_of_images,    
                  __global float * restrict output) {

int i,j,k;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 14 * 14;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <224; layer++){
	float temp_conv_val = bias[layer];
      
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            for(int d=0;d<112;d++){
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
                            temp_conv_val  += img[(image_number*14*14*112)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*112)+(d*3*3)+  (filterX*3)+filterY] ;
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

__kernel void Mixed_4c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
              __global float * restrict weights, 
              __global float * restrict bias,
               int number_of_images,     
              __global float * restrict output){
  int i,j,k;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 14 * 14;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <64; layer++){
	float temp_conv_val = bias[layer];
      
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            for(int d=0;d<24;d++){
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
		}
            output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
            index++;
          }
        }
      
    }
  }
}

__kernel void Padding_Mixed_4c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 512) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))];
  }
}

__kernel void Mixed_4c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input0[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void Mixed_4c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4c_concat(__global float* restrict T_concat, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((448 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -87808)] : (float)((384 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? input1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -75264)] : (float)((160 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? input2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -31360)] : input3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))])));
  }
}


