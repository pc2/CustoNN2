
__kernel void Mul1_1547_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]), 0.000000e+00f);
    }
}


__kernel void  block1_unit_2_bt_v2_conv1_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[64];
    for( int j = 0; j < 64;++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 64; ++ff) {
        //local memory for weights
        float input_weight[256];
        for( int k = 0; k < 256; ++k){
            input_weight[k] = input1[((ff * 256) + k)];
        }
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
				#pragma unroll
                for (int rc = 0; rc < 256; ++rc) {
                    temp_1 += (input0[((((rc * 56) + yy) * 56) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
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

__kernel void  block1_unit_2_bt_v2_conv2_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[64];
    for( int j = 0; j < 64; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 64; ++ff) {
        //local memory for weights
        float input_weight[3*3*64];
        for( int k = 0; k < 3*3*64; ++k){
            input_weight[k] = input1[((ff * 3*3*64) + k)];
        }
        
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 64; ++rc) {
                    float temp_2 = 0.0;
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry) {
                        float temp_1 = 0.0;
#pragma unroll
                        for (int rx = 0; rx < 3; ++rx) {
                            temp_1 += (input0[((((((rc * 58) + yy) + ry) * 58) + xx) + rx)] * input_weight[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}


__kernel void  block1_unit_2_bt_v2_conv3_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[256];
    for( int j = 0; j < 256; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 256; ++ff) {
        //local memory for weights
        float input_weight[64];
        for( int k = 0; k < 64; ++k){
            input_weight[k] = input1[((ff * 64) + k)];
        }
        
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input2[ff];
                float temp_1 = 0.0;
				#pragma unroll
                for (int rc = 0; rc < 64; ++rc) {
                    temp_1 += (input0[((((rc * 56) + yy) * 56) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}



__kernel void block1_unit_2_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
    }
}
