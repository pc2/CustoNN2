
__kernel void Mul1_1898_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f);
 }
}




__kernel void  block4_unit_2_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
   //Local memory for Biases:
    __local  float input_bias[512];
    for(int b = 0; b < 512; b++){
        input_bias[b] = input2[b];
    }

  for (int ff = 0; ff < 512; ++ff) {
  
	//Local weights 
        float input_weights[2048];
        for(int m = 0 ; m < 2048 ;m++){
            input_weights[m] = input1[((ff * 2048) + m)];
        }

    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
		float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
        for (int rc = 0; rc < 2048; ++rc) {
          temp_1 += (input0[((((rc * 7) + yy) * 7) + xx)] * input_weights[(rc)]);
        }
				temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
				compute[((((ff * 7) + yy) * 7) + xx)] = temp_0;
      }
    }
  }
}


__kernel void P_block4_unit_2_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 41472; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
  }
}


__kernel void  block4_unit_2_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  
  __local  float input_bias[512];
    for(int b = 0; b < 512; b++){
        input_bias[b] = input2[b];
    }
  for (int ff = 0; ff < 512; ++ff) {
	//Local weights 
        float input_weights[3*3*512];
        for(int m = 0 ; m < 3*3*512 ; m++){
            input_weights[m] = input1[((ff * 3*3*512) + m)];
        }

    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
		float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
        for (int rc = 0; rc < 512; ++rc) {
		float temp_2 = 0.0;
                    #pragma unroll

          for (int ry = 0; ry < 3; ++ry) {
		   float temp_1 = 0.0;
                        #pragma unroll

            for (int rx = 0; rx < 3; ++rx) {
              temp_1 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
            }
			temp_2 += temp_1;

          }
		  temp_3 += temp_2;
        }
	temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
				compute[((((ff * 7) + yy) * 7) + xx)] = temp_0;
      }
    }
  }
}


__kernel void  block4_unit_2_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  
  __local  float input_bias[2048];
    for(int b = 0; b < 2048; b++){
        input_bias[b] = input2[b];
    }
  for (int ff = 0; ff < 2048; ++ff) {
	 //Local weights 
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }

    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
			float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
        for (int rc = 0; rc < 512; ++rc) {
          temp_1 += (input0[((((rc * 7) + yy) * 7) + xx)] * input_weights[(rc)]);
        }
		 temp_0 += temp_1;
				compute[((((ff * 7) + yy) * 7) + xx)] = temp_0;
      }
    }
  }
}

__kernel void block4_unit_2_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner ] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}
