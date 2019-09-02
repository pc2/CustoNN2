__kernel void Mixed_4f_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //local memory for biases
    __local float input_bias[256];
    for (int j = 0; j < 256; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 256; ++ff)
    {
        //local memory for weights
        float input_weight[528];
        for(int m = 0 ; m < 528 ;m++){
            input_weight[m] = input1[((ff * 528) + m)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 528; ++rc)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
#pragma unroll 2
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                
                    temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
                }
            }
            }
#pragma loop_coalesce
            for (int yy = 0; yy < 14; ++yy)
            {
                for (int xx = 0; xx < 14; ++xx)
                {
                    temp_out[yy][xx] += input_bias[ff];
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 14) + yy) * 14) + xx)] = temp_out[yy][xx];
                    
                }
                
            }
        
    }
    
}


__kernel void Mixed_4f_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //local memory for biases
    __local float input_bias[160];
    for (int j = 0; j < 160; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 160; ++ff)
    {
        //local memory for weights
        float input_weight[528];
        for(int m = 0 ; m < 528 ;m++){
            input_weight[m] = input1[((ff * 528) + m)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 528; ++rc)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
#pragma unroll 2
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
            }
                
            }
        }
#pragma loop_coalesce
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 14) + yy) * 14) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

__kernel void Padding_Mixed_4f_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40960; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4f_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[320];
    for(int b = 0; b < 320; b++){
        input_bias[b] = input2[b];
    }

    float l_input[16*16];
    for (int ff = 0; ff < 320; ++ff)
    {
        //local memory for weights
        float input_weight[3*3*160];
        for(int m = 0 ; m < 3*3*160 ; m++){
            input_weight[m] = input1[((ff * 3*3*160) + m)];
        }
        float temp_out[14][14];
		#pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
		#pragma unroll 2
        for (int rc = 0; rc < 160; ++rc)
        {
            for (int i = 0; i < 16*16; i++){
                l_input[i] = input0[16*16*rc+i];
            }
#pragma unroll 2
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = 0;
#pragma unroll
                for (int rx = 0; rx < 3; ++rx)
                {
                    temp_0 += l_input[(yy+0) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 0) * 3) + rx)];
                }
                temp_out[yy][xx] += temp_0;
                
                float temp_1 = 0;
#pragma unroll
                for (int rx = 0; rx < 3; ++rx)
                {
                    temp_1 += l_input[(yy+1) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 1) * 3) + rx)];
                }
                temp_out[yy][xx] += temp_1;
                
                float temp_2 = 0;
#pragma unroll
                for (int rx = 0; rx < 3; ++rx)
                {
                    temp_2 += l_input[(yy+2) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 2) * 3) + rx)];
                }
                temp_out[yy][xx] += temp_2;
            }
            }
        }
#pragma loop_coalesce
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.0;
                 compute[((((ff * 14) + yy) * 14) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

__kernel void Mixed_4f_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[32];
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[196];
    for (int ff = 0; ff < 32; ++ff)
    {
        //local memory for weights
        float input_weight[528];
        for(int m = 0 ; m < 528 ;m++){
            input_weight[m] = input1[((ff * 528) + m)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 528; ++rc)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
#pragma unroll 2
        for (int yy = 0; yy < 14; ++yy)
        {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
            }
            
        }
        }
#pragma loop_coalesce
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 14) + yy) * 14) + xx)] = temp_out[yy][xx];
            }
        }
    }
}


__kernel void Padding_Mixed_4f_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4f_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[128];
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[16*16];
    for (int ff = 0; ff < 128; ++ff)
    {
        //local memory for weights
        float input_weight[3*3*32];
        for(int m = 0 ; m < 3*3*32 ; m++){
            input_weight[m] = input1[((ff * 3*3*32) + m)];
        }
        float temp_out[14][14];
		#pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
		#pragma unroll 2
        for (int rc = 0; rc < 32; ++rc)
        {
            for (int i = 0; i < 16*16; i++){
                l_input[i] = input0[16*16*rc+i];
            }
            
#pragma unroll 2
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 14; ++xx)
                {
                    float temp_0 = 0;
                    
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_0 += l_input[(yy+0) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 0) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_0;
                    
                    
                    float temp_1 = 0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_1 += l_input[(yy+1) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 1) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_1;
                    
                    float temp_2 = 0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_2 += l_input[(yy+2) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 2) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_2;
                }
            }
        }
#pragma loop_coalesce
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 14) + yy) * 14) + xx)] = temp_out[yy][xx];
            }
        }
    }
    
}

__kernel void Mixed_4f_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 528; ++ax1)
    {
        for (int ax2 = 0; ax2 < 14; ++ax2)
        {
            for (int ax3 = 0; ax3 < 14; ++ax3)
            {
                tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input0[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_4f_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[128];
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }
    float l_input[196];
    for (int ff = 0; ff < 128; ++ff)
    {
        //local memory for weights
        float input_weight[528];
        for(int m = 0 ; m < 528 ;m++){
            input_weight[m] = input1[((ff * 528) + m)];
        }
        float temp_out[14][14];
		#pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 528; ++rc)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
            
#pragma unroll 2
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 14; ++xx)
                {
                    temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
                }
                
            }
        }
        #pragma loop_coalesce
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 14) + yy) * 14) + xx)] = temp_out[yy][xx];
            }
        }
    }
}


__kernel void Mixed_4f_concat(__global float *restrict T_transpose, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 163072; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((137984 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -137984)] : (float)((112896 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -112896)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
