__kernel void  Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1 , __global float* restrict input2) {
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

__kernel void Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
                            __global float * restrict weights, 
                            __global float * restrict bias,
                             int number_of_images,     
                            __global double * restrict output){
  
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
        
            double temp_conv_val = bias[layer];
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


__kernel void Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D(__global double * restrict img, 
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


