__kernel void MaxPool_5a_2x2_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 832; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 2; ++rv) {
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], input0[((((((((ax1 * 7) + ax2) * 2) + rv) * 7) + ax3) * 2) + rv1)]);
          }
        }
      }
    }
  }
}

__kernel void Mixed_5b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
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


__kernel void Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,__global float* restrict input2) {
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

__kernel void Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D(__global double * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global double * restrict output){
  
  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 7*7;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <128; layer++) {
      for(int d=0;d<32;d++){
        for (i = 0; i < 7; i+=1) {
          for (j = 0; j < 7; j+=1) {
        
            double temp_conv_val = bias[layer];
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 1;
            int paderY = j - 1;

             for(int filterX=0; filterX<3; filterX++){
              for(int filterY=0; filterY<3; filterY++){
                if(paderX<0||paderX>=7||paderY<0||paderY>=7){}else{
                  temp_conv_val  += img[(image_number*7*7*32)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*3*3*32)+(d*3*3)+(filterX*3)+filterY] ;
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

__kernel void Padding_Mixed_5b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 49) * 832) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49))];
  }
}

__kernel void Mixed_5b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 832; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (8 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (8 - rv1))) ? input0[(((((((ax1 * 7) + ax2) + rv) * 7) + ax3) + rv1) + -8)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,__global float* restrict input2 ) {
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

__kernel void Mixed_5b_concat(__global float* restrict T_concat, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((704 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832)) + -34496)] : (float)((576 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? input1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832)) + -28224)] : (float)((256 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? input2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832)) + -12544)] : input3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832))])));
  }
}



