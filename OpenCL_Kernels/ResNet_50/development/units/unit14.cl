__kernel void Mul1_1898_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f);
    }
}




__kernel void  block4_unit_2_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    float l_input[49];
    __local float input_bias[512];
    
    for( int j = 0; j < 512; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 512; ++ff) {
        
        //Local weights
        float input_weights[2048];
        for(int m = 0 ; m < 2048 ;m++){
            input_weights[m] = input1[((ff * 2048) + m)];
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
        for (int rc = 0; rc < 2048; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
#pragma unroll
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[yy * 7 + xx] * input_weights[rc]);
                }
                
            }
        }
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


__kernel void P_block4_unit_2_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 41472; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
    }
}

__kernel void  block4_unit_2_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    
    float l_input[9*9];
    __local float input_bias[512];
    
    for( int j = 0; j < 512; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 512; ++ff) {
        //Local weights
        float input_weights[3*3*512];
        for(int m = 0 ; m < 3*3*512 ; m++){
            input_weights[m] = input1[((ff * 3*3*512) + m)];
        }
        float temp_out[7][7];
#pragma unroll
        for (int l = 0; l < 7; l++ ){
#pragma unroll
            for (int j = 0; j < 7; j++){

                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; ++rc)
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
                    float temp_0 = 0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_0 += l_input[(yy+0) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_0;
                    
                    
                    float temp_1 = 0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_1 += l_input[(yy+1) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_1;
                    
                    float temp_2 = 0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_2 += l_input[(yy+2) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_2;
                    
                }
            }
        }
#pragma unroll
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



__kernel void  block4_unit_2_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    
    float l_input[49];
    __local float input_bias[2048];
    
    for( int j = 0; j < 2048; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 2048; ++ff) {
        //Local weights
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
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
        for (int rc = 0; rc < 512; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
#pragma unroll
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[yy * 7 + xx] * input_weights[rc]);
                }
                
            }
        }
        for (int yy = 0; yy < 7; ++yy)
        {
		#pragma unroll 
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

__kernel void block4_unit_2_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner ] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
    }
}


