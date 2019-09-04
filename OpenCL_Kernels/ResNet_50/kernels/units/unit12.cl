__kernel void  block3_unit_6_bt_v2_shortcut_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
#pragma unroll
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        float temp_0 = -3.402823e+38f;
	tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(temp_0, input0[(((((ax1 * 7) + ax2) * 14) + ax3) * 2)]);
      }
    }
  }
}



__kernel void  Mul1_1844_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]), 0.000000e+00f);
 }
}



__kernel void  block3_unit_6_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,__global float* restrict input2) {
  float l_input[14*14];
    __local float input_bias[256];
    
    for( int j = 0; j < 256; ++j){
        input_bias[j] = input2[j];
    }
  for (int ff = 0; ff < 256; ++ff) {
    float input_weights[1024];
    for(int w = 0 ; w < 1024 ;w++){
      input_weights[w] = input1[((ff * 1024) + w)];
    }
    float temp_out[14][14];
#pragma unroll
    for (int l = 0; l < 14; l++ ){
#pragma unroll
      for (int j = 0; j < 14; j++){
        temp_out[l][j] = 0;
      }
    }
    for (int rc = 0; rc < 1024; ++rc) {
      for (int i = 0; i < 14*14; i++){
        l_input[i] = input0[14*14*rc+i];
      }
      #pragma unroll 
      for (int yy = 0; yy < 14; ++yy) {
        #pragma unroll
        for (int xx = 0; xx < 14; ++xx) {
          temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weights[rc]);
        
        }
      }
    }
    #pragma unroll 
    for (int yy = 0; yy < 14; ++yy){
     #pragma unroll
     for (int xx = 0; xx < 14; ++xx){
        temp_out[yy][xx] += input_bias[ff];
        temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
        compute[((((ff * 14) + yy) * 14) + xx)] = temp_out[yy][xx];
      }
    }
  }
}


__kernel void P_block3_unit_6_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 65536; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
  }
}



__kernel void  block3_unit_6_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  float l_input[9*9];
    __local float input_bias[256];
    
    for( int j = 0; j < 256; ++j){
        input_bias[j] = input2[j];
    }
  for (int ff = 0; ff < 256; ++ff) {
    float input_weights[256*3*3];
    for(int w = 0 ; w < 256*3*3 ;w++){
      input_weights[w] = input1[((ff * 256*3*3) + w)];
    }
    float temp_out[7][7];
#pragma unroll
    for (int l = 0; l < 7; l++ ){
#pragma unroll
      for (int j = 0; j < 7; j++){

        temp_out[l][j] = 0;
      }
    }
#pragma unroll 2
    for (int rc = 0; rc < 256; ++rc) {
      for (int i = 0; i < 9*9; i++){
        l_input[i] = input0[9*9*rc+i];
      }
      #pragma unroll
      for (int yy = 0; yy < 7; ++yy) {
        #pragma unroll
        for (int xx = 0; xx < 7; ++xx) {
           float temp_0 = 0;
           #pragma unroll
           for (int rx = 0; rx < 3; ++rx) {
               temp_0 += l_input[(yy+0) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
          }
          temp_out[yy][xx] += temp_0;
          float temp_1 = 0;
          #pragma unroll
           for (int rx = 0; rx < 3; ++rx) {
               temp_1 += l_input[(yy+1) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
          }
          temp_out[yy][xx] += temp_1;
          float temp_2 = 0;
          #pragma unroll
           for (int rx = 0; rx < 3; ++rx) {
               temp_2 += l_input[(yy+2) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
          }
          temp_out[yy][xx] += temp_2;

        }
      }
    }
    #pragma unroll
    for (int yy = 0; yy < 7; ++yy){
      #pragma unroll
      for (int xx = 0; xx < 7; ++xx){
        temp_out[yy][xx] += input_bias[ff];
        temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
        compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
      }
    }
  }
}




__kernel void  block3_unit_6_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  float l_input[7*7];
    __local float input_bias[1024];
    
    for( int j = 0; j < 1024; ++j){
        input_bias[j] = input2[j];
    }
  for (int ff = 0; ff < 1024; ++ff) {
    float input_weights[256];
    for(int w = 0 ; w < 256 ;w++){
      input_weights[w] = input1[((ff * 256) + w)];
    }
    float temp_out[7][7];
#pragma unroll
    for (int l = 0; l < 7; l++ ){
#pragma unroll
      for (int j = 0; j < 7; j++){
        temp_out[l][j] = 0;
      }
    }
#pragma unroll 2
    for (int rc = 0; rc < 256; ++rc) {
      for (int i = 0; i < 7*7; i++){
        l_input[i] = input0[7*7*rc+i];
      }
      #pragma unroll
      for (int yy = 0; yy < 7; ++yy) {
        #pragma unroll
        for (int xx = 0; xx < 7; ++xx) {
          temp_out[yy][xx] += (l_input[yy * 7 + xx] * input_weights[rc]);
        
        }
      }
    }
    #pragma unroll
    for (int yy = 0; yy < 7; ++yy){
     #pragma unroll
     for (int xx = 0; xx < 7; ++xx){
        temp_out[yy][xx] += input_bias[ff];
        compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
      }
    }
  }
}


__kernel void  block3_unit_6_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] =  input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

