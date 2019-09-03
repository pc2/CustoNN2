
__kernel void  block2_unit_4_bt_v2_shortcut_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
    for (int ax1 = 0; ax1 < 512; ++ax1) {
#pragma unroll
        for (int ax2 = 0; ax2 < 14; ++ax2) {
#pragma unroll
            for (int ax3 = 0; ax3 < 14; ++ax3) {
                tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
                tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], input0[(((((ax1 * 14) + ax2) * 28) + ax3) * 2)]);
            }
        }
    }
}


//56 //57

__kernel void Mul1_1682_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f);
    }
}



//58
__kernel void  block2_unit_4_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    
    float l_input[784];
    for (int ff = 0; ff < 128; ++ff) {
        float input_weights[512];

        for(int w = 0 ; w < 512 ;++w)
        {
            input_weights[w] = input1[((ff * 512) + w)];
        }
        float temp_out[28][28];

        for (int l = 0; l < 28; l++ ){

            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; rc++)
        {
            for (int i = 0; i < 28*28; i++){
                l_input[i] = input0[28*28*rc+i];
            }
            
#pragma unroll 4
            for (int yy = 0; yy < 28; ++yy)
            {
#pragma unroll 
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += (l_input[yy * 28 + xx] * input_weights[rc]);
                }
                
            }
        }
        #pragma unroll 4
        for (int yy = 0; yy < 28; ++yy)
        {
#pragma unroll
            for (int xx = 0; xx < 28; ++xx)
            {
                temp_out[yy][xx] += input2[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 28) + yy) * 28) + xx)] = temp_out[yy][xx];
            }
        }
    }
}





//P kernel
__kernel void P_block2_unit_4_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
    }
}

//60
__kernel void  block2_unit_4_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    __local float input_bias[128];
    for (int b = 0; b < 128; ++b) {
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 128; ++ff) {
        float input_weights[3*3*128];
        
        for (int w = 0; w < 3*3*128; ++w) {
            input_weights[w] = input1[((ff * 3*3*128) + w)];
        }
        for (int yy = 0; yy < 14; ++yy) {
            for (int xx = 0; xx < 14; ++xx) {
                float temp_3 = 0.0;
                float temp_0 = input_bias[ff];
                for (int rc = 0; rc < 128; ++rc) {
                    float temp_2 = 0.0;
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry) {
                        float temp_1 = 0.0;
#pragma unroll
                        for (int rx = 0; rx < 3; ++rx) {
                            temp_1 += (input0[((((((((rc * 15) + yy) * 2) + ry) * 15) + xx) * 2) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                compute[((((ff * 14) + yy) * 14) + xx)] = temp_0;
            }
        }
    }
}



__kernel void  block2_unit_4_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    float l_input[196];
    for (int ff = 0; ff < 512; ++ff) {
        float input_weights[128];
        for(int w = 0 ; w < 128 ;++w)
        {
            input_weights[w] = input1[((ff * 128) + w)];
        }
        
        float temp_out[14][14];
#pragma unroll
        for (int l = 0; l < 14; l++ ){
#pragma unroll
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 128; rc++)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
            
#pragma unroll 
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll 
                for (int xx = 0; xx < 14; ++xx)
                {
                    temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weights[rc]);
                }
                
            }
        }
        #pragma unroll
        for (int yy = 0; yy < 14; ++yy)
        {
            #pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input2[ff];
                compute[((((ff * 14) + yy) * 14) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

//63
__kernel void block2_unit_4_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner ] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
    }
}


