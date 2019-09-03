__kernel void Mul1_1547_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]), 0.000000e+00f);
  }
}


__kernel void  block1_unit_2_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  float l_input[56*56];
  for (int ff = 0; ff < 64; ++ff) {
    float input_weights[256];
    for(int w = 0 ; w < 256 ;w++){
      input_weights[w] = input1[((ff * 256) + w)];
    }
    float temp_out[56][56];
    for (int l = 0; l < 56; l++ ){
      for (int j = 0; j < 56; j++){
        temp_out[l][j] = 0;
      }
    }
    for (int rc = 0; rc < 256; ++rc) {
      for (int i = 0; i < 56*56; i++){
        l_input[i] = input0[56*56*rc+i];
      }
#pragma unroll 4
      for (int yy = 0; yy < 56; ++yy) {
#pragma unroll 
        for (int xx = 0; xx < 56; ++xx) {
	  temp_out[yy][xx] += (l_input[yy * 56 + xx] * input_weights[rc]);
        }
      }
    }

    for (int yy = 0; yy < 56; ++yy){
     for (int xx = 0; xx < 56; ++xx){
        temp_out[yy][xx] += input2[ff];
        temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
        compute[((((ff * 56) + yy) * 56) + xx)] = temp_out[yy][xx];
      }
    }
  }
}


__kernel void P_block1_unit_2_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f);
    }

}

__kernel void  block1_unit_2_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  float l_input[58*58];
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
      #pragma unroll 4
      for (int yy = 0; yy < 56; ++yy) {
        #pragma unroll 
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
        temp_out[yy][xx] += input2[ff];
        temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
        compute[((((ff * 56) + yy) * 56) + xx)] = temp_out[yy][xx];
      }
    }
  }
}


__kernel void  block1_unit_2_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  float l_input[56*56];
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
#pragma unroll 4
      for (int yy = 0; yy < 56; ++yy) {
#pragma unroll 
        for (int xx = 0; xx < 56; ++xx) {
	  temp_out[yy][xx] += (l_input[yy * 56 + xx] * input_weights[rc]);

        }
      }
    }

    for (int yy = 0; yy < 56; ++yy){
     for (int xx = 0; xx < 56; ++xx){
        temp_out[yy][xx] += input2[ff];
        compute[((((ff * 56) + yy) * 56) + xx)] = temp_out[yy][xx];
      }
    }
  }
}



__kernel void block1_unit_2_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}


