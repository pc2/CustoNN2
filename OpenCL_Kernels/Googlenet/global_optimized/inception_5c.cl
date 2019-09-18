__kernel void Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{//Local memory for Biases:
    __local  float input_bias[384];
    for(int b = 0; b < 384; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[49];
    for (int ff = 0; ff < 384; ++ff)
    {
        //Local weights
        float input_weights[832];
        for(int m = 0 ; m < 832 ;m++){
            input_weights[m] = input1[((ff * 832) + m)];
        }
        
        float temp_out[7][7];
#pragma loop_coalesce
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0.0;
            }
        }
        for (int rc = 0; rc < 832; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            
#pragma unroll 2
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[ yy * 7 + xx] * input_weights[rc]);
                }
                
                
            }
        }
#pragma loop_coalesce
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

__kernel void Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[192];
    for(int b = 0; b < 192; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[49];
    for (int ff = 0; ff < 192; ++ff)
    {
        //Local weights
        float input_weights[832];
        for(int m = 0 ; m < 832 ;m++){
            input_weights[m] = input1[((ff * 832) + m)];
        }
        
        float temp_out[7][7];
#pragma loop_coalesce
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0.0;
            }
        }
        for (int rc = 0; rc < 832; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            
#pragma unroll 2
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[ yy * 7 + xx] * input_weights[rc]);
                }
                
                
            }
        }
#pragma loop_coalesce
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

__kernel void Padding_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 15552; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
    }
}
__kernel void Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[384];
    for(int b = 0; b < 384; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[9*9];
    for (int ff = 0; ff < 384; ++ff)
    {
        //Local weights
        float input_weights[192*3*3];
        for(int m = 0 ; m < 192*3*3 ; m++){
            input_weights[m] = input1[((ff * 192*3*3) + m)];
        }
        float temp_out[7][7];
#pragma loop_coalesce
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 192; ++rc)
        {
            for (int i = 0; i < 9*9; i++){
                l_input[i] = input0[9*9*rc+i];
            }
#pragma unroll
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    float temp_0 = 0.0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_0 += l_input[(yy+0) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_0;
                    
                    float temp_1 = 0.0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_1 += l_input[(yy+1) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_1;
                    
                    
                    float temp_2 = 0.0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_2 += l_input[(yy+2) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_2;
                }
            }
        }
        for (int yy = 0; yy < 7; ++yy)
        {
#pragma unroll
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

__kernel void Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[48];
    for(int b = 0; b < 48; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[49];
    for (int ff = 0; ff < 48; ++ff)
    {
        //Local weights
        float input_weights[832];
        for(int m = 0 ; m < 832 ;m++){
            input_weights[m] = input1[((ff * 832) + m)];
        }
        
        float temp_out[7][7];
#pragma loop_coalesce
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0.0;
            }
        }
        for (int rc = 0; rc < 832; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            
#pragma unroll 2
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[ yy * 7 + xx] * input_weights[rc]);
                }
                
                
            }
        }
#pragma loop_coalesce
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

__kernel void Padding_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 3888; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
    }
}
__kernel void Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[128];
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[9*9];
    for (int ff = 0; ff < 128; ++ff)
    {
        //Local weights
        float input_weights[48*3*3];
        for(int m = 0 ; m < 48*3*3 ; m++){
            input_weights[m] = input1[((ff * 48*3*3) + m)];
        }
        float temp_out[7][7];
#pragma loop_coalesce
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
#pragma unroll 2
        for (int rc = 0; rc < 48; ++rc)
        {
            for (int i = 0; i < 9*9; i++){
                l_input[i] = input0[9*9*rc+i];
            }
#pragma unroll
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    float temp_0 = 0.0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_0 += l_input[(yy+0) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_0;
                    
                    float temp_1 = 0.0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_1 += l_input[(yy+1) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_1;
                    
                    
                    float temp_2 = 0.0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_2 += l_input[(yy+2) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_2;
                }
            }
        }
        for (int yy = 0; yy < 7; ++yy)
        {
#pragma unroll
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

__kernel void Mixed_5c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 832; ++ax1)
    {
        for (int ax2 = 0; ax2 < 7; ++ax2)
        {
            for (int ax3 = 0; ax3 < 7; ++ax3)
            {
                tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (8 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (8 - rv1))) ? input0[(((((((ax1 * 7) + ax2) + rv) * 7) + ax3) + rv1) + -8)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[128];
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[49];
    for (int ff = 0; ff < 128; ++ff)
    {
        //Local weights
        float input_weights[832];
        for(int m = 0 ; m < 832 ;m++){
            input_weights[m] = input1[((ff * 832) + m)];
        }
        
        float temp_out[7][7];
#pragma loop_coalesce
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0.0;
            }
        }
        for (int rc = 0; rc < 832; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            
#pragma unroll 2
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[ yy * 7 + xx] * input_weights[rc]);
                }
                
                
            }
        }
#pragma loop_coalesce
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

__kernel void Mixed_5c_concat(__global float *restrict T_transpose, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((43904 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -43904)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : (float)((18816 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -18816)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}

__kernel void AvgPool_0a_7x7_AvgPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 1024; ++ax1)
    {
        tensor[ax1] = 0.000000e+00f;
        for (int rv = 0; rv < 7; ++rv)
        {
            for (int rv1 = 0; rv1 < 7; ++rv1)
            {
                tensor[ax1] = (tensor[ax1] + (input0[((((ax1 * 7) + rv) * 7) + rv1)] * 2.040816e-02f));
            }
        }
    }
}

__kernel void Conv2d_0c_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    __local  float input_bias[1001];
    for(int b = 0; b < 1001; b++){
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 1001; ++ff)
    {
        float input_weights[1024];
        for(int w = 0; w < 1024; w++){
            input_weights[w] = input1[((ff * 1024) + w)];
        }
        
        compute[ff] = input_bias[ff];
        float temp_1 = 0.0;
        for (int rc = 0; rc < 1024; ++rc)
        {
            temp_1 += (input0[rc] * input_weights[rc]);
        }
        compute[ff] += temp_1;
        compute[ff] = (compute[ff] > 0) ? compute[ff] : 0.0;
    }
}



