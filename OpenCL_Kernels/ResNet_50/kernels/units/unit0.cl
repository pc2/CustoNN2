__kernel void  Mul1__Fused_Mul__FusedScaleShift(__global float* restrict T_relu, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_relu[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = ((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50176)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50176)]);
  }
}


__kernel void  P_conv1_Conv2D(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 158700; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((690 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52900) < 52210)) && (3 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 230))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 230) < 227)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52900) / 230) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 230)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52900)) + -2025)] : 0.000000e+00f);
  }
}

__kernel void  conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
   //Local memory for Biases:
   __local  float input_bias[64];
   for(int b = 0; b < 64; b++){
       input_bias[b] = input2[b];
   }
   for (int ff = 0; ff < 64; ++ff)
   {
       //Local weights
       float  local_weight[7*7*3];
       for(int m = 0 ; m < 7*7*3 ;m++){
           local_weight[m] = input1[((ff * 7*7*3) + m)];
       }
       //2D array to store Temporary results of 1 slice.
       float temp_out[112][112];
       //Initialize values with 0
       #pragma loop_coalesce
       for (int l = 0; l < 112; l++ ){
           for (int j = 0; j < 112; j++){
               temp_out[l][j] = 0;
           }
       }
       for (int rc = 0; rc < 3; ++rc)
       {
           //Store 1 slice of input image
            float image_slice[230*230];
           for (int in = 0; in < 230*230; in++){
                   image_slice[in]= input0[(230*230*rc)+in];
           }
           for (int yy = 0; yy < 112; ++yy)
           {
               //#pragma unroll 4
               for (int xx = 0; xx < 112; ++xx)
               {
                   float temp_0 = 0;
                   float temp_2 = 0;
                       //#pragma unroll
                       for (int ry = 0; ry < 7; ++ry)
                       {
                           float temp_1 = 0;
                           //#pragma unroll
                           for (int rx = 0; rx < 7; ++rx)
                           {
                               temp_1 +=  (image_slice[(yy * 460) + (ry * 230) + (xx*2) + rx] * local_weight[(((((rc) * 7) + ry) * 7) + rx)]);
                           }
                           temp_2 +=temp_1;
                       }
                       temp_0 += temp_2;
                       temp_out[yy][xx] += temp_0;
               }
           }
       }
       //Summarize the results depthwise.
            #pragma loop_coalesce
           for (int yy = 0; yy < 112; ++yy)
           {
               for (int xx = 0; xx < 112; ++xx)
               {
                   temp_out[yy][xx] += input_bias[ff];
                   //RELU
                   //temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                   compute[((((ff * 112) + yy) * 112) + xx)] = temp_out[yy][xx];
            }
        }
    }
}


__kernel void  pool1_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
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



__kernel void  Mul1_1520_Fused_Mul__FusedScaleShift(__global float* restrict T_relu, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_relu[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]), 0.000000e+00f);
  }
}





__kernel void  block1_unit_1_bt_v2_shortcut_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  float l_input[56*56];
    __local float input_bias[256];
    
    for( int j = 0; j < 256; ++j){
        input_bias[j] = input2[j];
    }
    
  for (int ff = 0; ff < 256; ++ff) {
    float input_weights[64];
    for(int w = 0 ; w < 64 ;w++){
      input_weights[w] = input1[((ff * 64) + w)];
    }
    float temp_out[56][56];
    for (int l = 0; l < 56; l++ ){
      for (int j = 0; j < 56; j++){
        temp_out[l][j] = 0;
      }
    }
    for (int rc = 0; rc < 64; ++rc) {
      for (int i = 0; i < 56*56; i++){
        l_input[i] = input0[56*56*rc+i];
      }
#pragma unroll 2
      for (int yy = 0; yy < 56; ++yy) {
#pragma unroll 
        for (int xx = 0; xx < 56; ++xx) {
	  temp_out[yy][xx] += (l_input[yy * 56 + xx] * input_weights[rc]);
        }
      }
    }

    for (int yy = 0; yy < 56; ++yy){

     for (int xx = 0; xx < 56; ++xx){
        temp_out[yy][xx] += input_bias[ff];
        compute[((((ff * 56) + yy) * 56) + xx)] = temp_out[yy][xx];
      }
    }
  }
}


__kernel void  block1_unit_1_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  float l_input[56*56];
    __local float input_bias[64];
    
    for( int j = 0; j < 64; ++j){
        input_bias[j] = input2[j];
    }
    
  for (int ff = 0; ff < 64; ++ff) {
    float input_weights[64];
    for(int w = 0 ; w < 64 ;w++){
      input_weights[w] = input1[((ff * 64) + w)];
    }
    float temp_out[56][56];
    for (int l = 0; l < 56; l++ ){
      for (int j = 0; j < 56; j++){
        temp_out[l][j] = 0;
      }
    }
    for (int rc = 0; rc < 64; ++rc) {
      for (int i = 0; i < 56*56; i++){
        l_input[i] = input0[56*56*rc+i];
      }
// #pragma unroll 2
      for (int yy = 0; yy < 56; ++yy) {
#pragma unroll
        for (int xx = 0; xx < 56; ++xx) {
	  temp_out[yy][xx] += (l_input[yy * 56 + xx] * input_weights[rc]);

        }

      }
    }

    for (int yy = 0; yy < 56; ++yy){
     for (int xx = 0; xx < 56; ++xx){
        temp_out[yy][xx] += input_bias[ff];
        temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
        compute[((((ff * 56) + yy) * 56) + xx)] = temp_out[yy][xx];
      }
    }
  }
}


__kernel void P_block1_unit_1_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f);
    }
}


__kernel void  block1_unit_1_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  float l_input[58*58];
    
    __local float input_bias[64];
    
    for( int j = 0; j < 64; ++j){
        input_bias[j] = input2[j];
    }
    
  for (int ff = 0; ff < 64; ++ff) {
    float input_weights[64*3*3];
    for(int w = 0 ; w < 64*3*3 ;w++){
      input_weights[w] = input1[((ff * 64*3*3) + w)];
    }
    float temp_out[56][56];
    for (int l = 0; l < 56; l++ ){
      for (int j = 0; j < 56; j++){
        temp_out[l][j] = 0;
      }
    }
    for (int rc = 0; rc < 64; ++rc) {
      for (int i = 0; i < 58*58; i++){
        l_input[i] = input0[58*58*rc+i];
      }
 // #pragma unroll 2     
      for (int yy = 0; yy < 56; ++yy) {
        #pragma unroll 28
        for (int xx = 0; xx < 56; ++xx) {
          float temp_0 = 0;
           #pragma unroll
           for (int rx = 0; rx < 3; ++rx) {
               temp_0 += l_input[(yy+0) * 58 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
          }
          temp_out[yy][xx] += temp_0;
          float temp_1 = 0;
          #pragma unroll
           for (int rx = 0; rx < 3; ++rx) {
               temp_1 += l_input[(yy+1) * 58+ xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
          }
          temp_out[yy][xx] += temp_1;
          float temp_2 = 0;
          #pragma unroll
           for (int rx = 0; rx < 3; ++rx) {
               temp_2 += l_input[(yy+2) * 58 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
          }
          temp_out[yy][xx] += temp_2;
        }

      }
    }

    for (int yy = 0; yy < 56; ++yy){
     for (int xx = 0; xx < 56; ++xx){
        temp_out[yy][xx] += input_bias[ff];
        temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
        compute[((((ff * 56) + yy) * 56) + xx)] = temp_out[yy][xx];
      }
    }
  }
}



__kernel void  block1_unit_1_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  float l_input[56*56];
    
    __local float input_bias[256];
    
    for( int j = 0; j < 256; ++j){
        input_bias[j] = input2[j];
    }
    
  for (int ff = 0; ff < 256; ++ff) {
    float input_weights[64];
    for(int w = 0 ; w < 64 ;w++){
      input_weights[w] = input1[((ff * 64) + w)];
    }
    float temp_out[56][56];
    for (int l = 0; l < 56; l++ ){
      for (int j = 0; j < 56; j++){
        temp_out[l][j] = 0;
      }
    }
    for (int rc = 0; rc < 64; ++rc) {
      for (int i = 0; i < 56*56; i++){
        l_input[i] = input0[56*56*rc+i];
      }
#pragma unroll 2
      for (int yy = 0; yy < 56; ++yy) {
#pragma unroll 
        for (int xx = 0; xx < 56; ++xx) {
	  temp_out[yy][xx] += (l_input[yy * 56 + xx] * input_weights[rc]);

        }
      }
    }
    for (int yy = 0; yy < 56; ++yy){

     for (int xx = 0; xx < 56; ++xx){
        temp_out[yy][xx] += input_bias[ff];
        compute[((((ff * 56) + yy) * 56) + xx)] = temp_out[yy][xx];
      }
    }
  }
}



__kernel void  block1_unit_1_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] =  input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}


