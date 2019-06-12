__kernel void InceptionV1_Conv2d_1a_7x7_Conv2D(__global double * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global double * restrict output){

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
        
            double temp_conv_val = bias[layer];
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


__kernel void InceptionV1_MaxPool_2a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
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

__kernel void InceptionV1_Conv2d_2b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
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


__kernel void InceptionV1_Conv2d_2c_3x3_Conv2D(__global double * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias,
         int number_of_images,     
        __global double * restrict output){
  
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
            double temp_conv_val = bias[layer];
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

__kernel void InceptionV1_MaxPool_3a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
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

//MAXPOOL 4A Missing

__kernel void InceptionV1_MaxPool_5a_2x2_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
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


__kernel void InceptionV1_Logits_AvgPool_0a_7x7_AvgPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    tensor[ax1] = 0.000000e+00f;
    for (int rv = 0; rv < 7; ++rv) {
      for (int rv1 = 0; rv1 < 7; ++rv1) {
        tensor[ax1] = (tensor[ax1] + (input0[((((ax1 * 7) + rv) * 7) + rv1)] * 2.040816e-02f));
      }
    }
  }
}


__kernel void InceptionV1_Logits_Conv2d_0c_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,  __global float* restrict input2) {
  for (int ff = 0; ff < 1001; ++ff) {
    compute[ff] = input2[ff];
    for (int rc = 0; rc < 1024; ++rc) {
      compute[ff] = (compute[ff] + (input0[rc] * input1[((ff * 1024) + rc)]));
    }
  }
}

// TODO InceptionV1/Logits/Conv2d_0c_1x1/Conv2D/Permute_


__kernel void InceptionV1_Logits_Predictions_Reshape(__global float* restrict tensor, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    tensor[ax0_ax1_fused_inner] = (input0[ax0_ax1_fused_inner] + input1[ax0_ax1_fused_inner]);
  }
}

__kernel void InceptionV1_Logits_Predictions_Softmax(__global float* restrict tensor, __global float* restrict input0, __global float* restrict tensor1, __global float* restrict tensor2) {
  for (int ax1 = 0; ax1 < 1001; ++ax1) {
    tensor[0] = -3.402823e+38f;
    for (int k1 = 0; k1 < 1001; ++k1) {
      tensor[0] = max(tensor[0], input0[k1]);
    }
    tensor1[0] = 0.000000e+00f;
    for (int k2 = 0; k2 < 1001; ++k2) {
      tensor1[0] = (tensor1[0] + exp((input0[k2] - tensor[0])));
    }
    tensor2[ax1] = (exp((input0[ax1] - tensor[0])) / tensor1[0]);
  }
}

__kernel void InceptionV1_Logits_Predictions_Reshape_1(__global float* restrict T_reshape, __global float* restrict input0) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    T_reshape[ax0_ax1_fused_inner] = input0[ax0_ax1_fused_inner];
  }
}














