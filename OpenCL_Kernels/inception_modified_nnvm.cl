__kernel void Conv2d_1a_7x7_Conv2D(__global float * restrict img, 
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
      for(int d=0;d<3;d++){
        for (i = 0; i < 224; i+=2){
          for (j = 0; j < 224; j+=2){
        
            float temp_conv_val = bias[layer];
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 2;
            int paderY = j - 2;

             for(int filterX=0; filterX<7; filterX++){
                    for(int filterY=0; filterY<7; filterY++){
                if(paderX<0||paderX>=224||paderY<0||paderY>=224){}else{
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
            output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
            index++;
          }
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
    for (layer = 0; layer <192; layer++) {
      for(int d=0;d<64;d++){
        for (i = 0; i < 56; i+=1) {
          for (j = 0; j < 56; j+=1) {
            float temp_conv_val = bias[layer];
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
            output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
            index++;
          }
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

__kernel void Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
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

__kernel void Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1 , __global float* restrict input2) {
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


__kernel void Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global float * restrict output){
  
  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 28*28;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <128; layer++) {
      for(int d=0;d<96;d++){
        for (i = 0; i < 28; i+=1){
          for (j = 0; j < 28; j+=1) {
        
            float temp_conv_val = bias[layer];
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 1;
            int paderY = j - 1;

             for(int filterX=0; filterX<3; filterX++){
                for(int filterY=0; filterY<3; filterY++){
                  if(paderX<0||paderX>=28||paderY<0||paderY>=28){}else{
                    temp_conv_val  += img[(image_number*28*28*96)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*96)+(d*3*3)+(filterX*3)+filterY] ;
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

__kernel void Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
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

__kernel void Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global float * restrict output){
  
  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 28*28;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <32; layer++){
      for(int d=0;d<16;d++){
        for (i = 0; i < 28; i+=1){
          for (j = 0; j < 28; j+=1){
        
            float temp_conv_val = bias[layer];
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 1;
            int paderY = j - 1;

            for(int filterX=0; filterX<3; filterX++){
                for(int filterY=0; filterY<3; filterY++){
                  if(paderX<0||paderX>=28||paderY<0||paderY>=28){}else{
                    temp_conv_val  += img[(image_number*28*28*16)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*16)+(d*3*3)+(filterX*3)+filterY] ;
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

__kernel void Padding_Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 192) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))];
  }
}

__kernel void Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 192; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? input0[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}


__kernel void Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1 , __global float* restrict input2) {
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

__kernel void Mixed_3b_concat(__global float* restrict T_concat, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(224 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -175616)] : (float)(192 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) ? input1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -150528)] : (float)(64 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) ? input2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -50176)] : input3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256))])));
  }
}


__kernel void Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1 , __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff){
    for (int yy = 0; yy < 28; ++yy){
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

__kernel void Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1 , __global float* restrict input2) {
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

__kernel void Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
                            __global float * restrict weights, 
                            __global float * restrict bias,
                             int number_of_images,     
                            __global float * restrict output){
  
  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 28*28;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <192; layer++) {
      for(int d=0;d<128;d++){
        for (i = 0; i < 28; i+=1){
          for (j = 0; j < 28; j+=1){
        
            float temp_conv_val = bias[layer];
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 1;
            int paderY = j - 1;

            for(int filterX=0; filterX<3; filterX++){
              for(int filterY=0; filterY<3; filterY++){
                if(paderX<0||paderX>=28||paderY<0||paderY>=28){}else{
                  temp_conv_val  += img[(image_number*28*28*128)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*128)+(d*3*3)+(filterX*3)+filterY] ;
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


__kernel void Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
                  __global float* restrict input0, 
                  __global float* restrict input1,
                  __global float* restrict input2 ){
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


__kernel void Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
                  __global float * restrict weights, 
                  __global float * restrict bias,
                  int number_of_images,    
                  __global float * restrict output) {

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
            float temp_conv_val = bias[layer];
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 1;
            int paderY = j - 1;

             for(int filterX=0; filterX<3; filterX++){
                for(int filterY=0; filterY<3; filterY++){
                  if(paderX<0||paderX>=28||paderY<0||paderY>=28){}else{
                    temp_conv_val  += img[(image_number*28*28*32)+(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*32)+(d*3*3)+    (filterX*3)+filterY] ;
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

__kernel void Mixed_3c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? input0[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}


__kernel void Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, 
                  __global float* restrict input0, 
                  __global float* restrict input1,
                  __global float* restrict input2 ){
  

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


__kernel void Mixed_3c_concat(__global float* restrict T_transpose, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 376320; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float) ((326144 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input0[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -326144)] : (float)((250880 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -250880)] : (float)((100352 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -100352)] : input3[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
  }
}


__kernel void MaxPool_4a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 480; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((ax2 * 2) < (28 - rv)) && ((ax3 * 2) < (28 - rv1))) ? input0[((((((((ax1 * 14) + ax2) * 2) + rv) * 14) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}




__kernel void Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
                  __global float* restrict input0, 
                  __global float* restrict input1,
                  __global float* restrict input2 ){
  

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

__kernel void Mixed_4b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
                  __global float* restrict input0, 
                  __global float* restrict input1,
                  __global float* restrict input2 ){
  

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


__kernel void Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
                  __global float * restrict weights, 
                  __global float * restrict bias,
                  int number_of_images,    
                  __global float * restrict output) {

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
            float temp_conv_val = bias[layer];
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
                            temp_conv_val  += img[(image_number*14*14*96)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*96)+(d*3*3)+    (filterX*3)+filterY] ;
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


__kernel void Mixed_4b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
                  __global float* restrict input0, 
                  __global float* restrict input1,
                  __global float* restrict input2 ){
  

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


__kernel void Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
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
    for (layer = 0; layer <48; layer++){
      for(int d=0;d<16;d++){
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            float temp_conv_val = bias[layer];
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
                            temp_conv_val  += img[(image_number*14*14*16)+(d*14*14)+(14*PaddedX)+PaddedY] *weights[(layer*3*3*16)+(d*3*3)+    (filterX*3)+filterY] ;
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

__kernel void Padding_Mixed_4b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 94080; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 480) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))];
  }
}

__kernel void Mixed_4b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 480; ++ax1) {
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

__kernel void Mixed_4b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, 
                  __global float* restrict input0, 
                  __global float* restrict input1,
                  __global float* restrict input2 ){
  

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


__kernel void Mixed_4b_concat(__global float* restrict T_concat, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((448 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -87808)] : (float)((400 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? input1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -78400)] : (float)((192 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? input2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -37632)] : input3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))])));
  }
}


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
      for(int d=0;d<112;d++){
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            float temp_conv_val = bias[layer];
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
            output[index] = (temp_conv_val>0) ? temp_conv_val : 0;
            index++;
          }
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
      for(int d=0;d<24;d++){
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            float temp_conv_val = bias[layer];
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



__kernel void Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4d_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
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
    for (layer = 0; layer <256; layer++){
      for(int d=0;d<128;d++){
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            float temp_conv_val = bias[layer];
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

__kernel void Mixed_4d_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
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
      for(int d=0;d<24;d++){
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            float temp_conv_val = bias[layer];
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


__kernel void Padding_Mixed_4d_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 512) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))];
  }
}

__kernel void Mixed_4d_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
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


__kernel void Mixed_4d_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, 
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


__kernel void Mixed_4d_concat(__global float* restrict T_concat, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((448 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -87808)] : (float)((384 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? input1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -75264)] : (float)((128 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? input2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -25088)] : input3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))])));
  }
}


__kernel void Mixed_4e_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4e_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
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
    for (layer = 0; layer <288; layer++){
      for(int d=0;d<144;d++){
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            float temp_conv_val = bias[layer];
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

__kernel void Mixed_4e_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
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
      for(int d=0;d<32;d++){
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            float temp_conv_val = bias[layer];
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


__kernel void Padding_Mixed_4e_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 512) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))];
  }
}

__kernel void Mixed_4e_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
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

__kernel void Mixed_4e_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4e_concat(__global float* restrict T_concat, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((464 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 528)) + -90944)] : (float)((400 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528)) ? input1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 528)) + -78400)] : (float)((112 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528)) ? input2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 528)) + -21952)] : input3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 528))])));
  }
}

__kernel void Mixed_4f_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4f_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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


__kernel void Mixed_4f_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
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
    for (layer = 0; layer <320; layer++){
      for(int d=0;d<160;d++){
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            float temp_conv_val = bias[layer];
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

__kernel void Mixed_4f_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4f_Branch_2_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
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
    for (layer = 0; layer <128; layer++){
      for(int d=0;d<32;d++){
        for (i = 0; i < 14; i+=1){
          for (j = 0; j < 14; j+=1){
            float temp_conv_val = bias[layer];
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


__kernel void Padding_Mixed_4f_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 528) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))];
  }
}

__kernel void Mixed_4f_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 528; ++ax1) {
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


__kernel void Mixed_4f_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, 
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

__kernel void Mixed_4f_concat(__global float* restrict T_transpose, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 163072; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((137984 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input0[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -137984)] : (float)((112896 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -112896)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input3[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
  }
}

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

__kernel void Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
                      __global float * restrict weights, 
                      __global float * restrict bias,
                       int number_of_images,     
                      __global float * restrict output){
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
            float temp_conv_val = bias[layer];
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

__kernel void Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D(__global float * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global float * restrict output){
  
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
        
            float temp_conv_val = bias[layer];
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


__kernel void Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
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

__kernel void Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,__global float* restrict input2) {
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


__kernel void Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global float * restrict output){
  
  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 7*7;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <384; layer++){
      for(int d=0;d<192;d++){
        for (i = 0; i < 7; i+=1) {
          for (j = 0; j < 7; j+=1) {
        
            float temp_conv_val = bias[layer];
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 1;
            int paderY = j - 1;

             for(int filterX=0; filterX<3; filterX++){
                for(int filterY=0; filterY<3; filterY++){
                if(paderX<0||paderX>=7||paderY<0||paderY>=7){}else{
                  temp_conv_val  += img[(image_number*7*7*192)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*3*3*192)+(d*3*3)+(filterX*3)+filterY] ;
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


__kernel void Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
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


__kernel void Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global float * restrict output){
  

  //printf("Inside conv layer\n");
  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  int image_number;
  index = 0;
  image_size = 7*7;
  for (image_number = 0; image_number < number_of_images; image_number++){
    for (layer = 0; layer <128; layer++) {
      for(int d=0;d<128;d++){
        for (i = 0; i < 7; i+=1){
          for (j = 0; j < 7; j+=1) {
        
            float temp_conv_val = bias[layer];
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 1;
            int paderY = j - 1;

             for(int filterX=0; filterX<3; filterX++){
              for(int filterY=0; filterY<3; filterY++){
                if(paderX<0||paderX>=7||paderY<0||paderY>=7){}else{
                  temp_conv_val  += img[(image_number*7*7*128)+(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*3*3*128)+(d*3*3)+(filterX*3)+filterY] ;
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

__kernel void Padding_Mixed_5c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 49) * 832) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49))];
  }
}

__kernel void Mixed_5c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
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

__kernel void Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,  __global float* restrict input2) {
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

__kernel void Mixed_5c_concat(__global float* restrict T_transpose, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((43904 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input0[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -43904)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : (float)((18816 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -18816)] : input3[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
  }
}

__kernel void Logits_AvgPool_0a_7x7_AvgPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    tensor[ax1] = 0.000000e+00f;
    for (int rv = 0; rv < 7; ++rv) {
      for (int rv1 = 0; rv1 < 7; ++rv1) {
        tensor[ax1] = (tensor[ax1] + (input0[((((ax1 * 7) + rv) * 7) + rv1)] * 2.040816e-02f));
      }
    }
  }
}


__kernel void Logits_Conv2d_0c_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,  __global float* restrict input2) {
  for (int ff = 0; ff < 1001; ++ff) {
    compute[ff] = input2[ff];
    for (int rc = 0; rc < 1024; ++rc) {
      compute[ff] = (compute[ff] + (input0[rc] * input1[((ff * 1024) + rc)]));
    }
  }
}

// TODO InceptionV1/Logits/Conv2d_0c_1x1/Conv2D/Permute_


__kernel void Logits_Predictions_Reshape(__global float* restrict tensor, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    tensor[ax0_ax1_fused_inner] = (input0[ax0_ax1_fused_inner] + input1[ax0_ax1_fused_inner]);
  }
}

__kernel void Logits_Predictions_Softmax(__global float* restrict input0, 
					__global float* restrict tensor2) {
  float tensor,tensor1;
  for (int ax1 = 0; ax1 < 1001; ++ax1) {
    tensor = -3.402823e+38f;
    for (int k1 = 0; k1 < 1001; ++k1) {
      tensor = max(tensor, input0[k1]);
    }
    tensor1[0] = 0.000000e+00f;
    for (int k2 = 0; k2 < 1001; ++k2) {
      tensor1 = (tensor1 + exp((input0[k2] - tensor)));
    }
    tensor2[ax1] = (exp((input0[ax1] - tensor)) / tensor1);
  }
}

__kernel void Logits_Predictions_Reshape_1(__global float* restrict T_reshape, __global float* restrict input0) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    T_reshape[ax0_ax1_fused_inner] = input0[ax0_ax1_fused_inner];
  }
}














