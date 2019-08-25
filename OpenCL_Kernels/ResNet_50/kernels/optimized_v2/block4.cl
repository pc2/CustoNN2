__kernel void  Mul1_1871_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f);
    }
}


//116

__kernel void  block4_unit_1_bt_v2_shortcut_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2)
{
    float l_input[49];
    for (int ff = 0; ff < 2048; ++ff)
    {
        //Local weights
        float input_weights[1024];
        for(int m = 0 ; m < 1024 ;m++){
            input_weights[m] = input1[((ff * 1024) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 1024; rc++)
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
                    temp_out[yy][xx] += (l_input[yy * 7 + xx] * input_weights[rc]);
                }
                
            }
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input2[ff];
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}

//117


__kernel void  block4_unit_1_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,__global float* restrict input2) {
    float l_input[49];
    for (int ff = 0; ff < 512; ++ff)
    {
        //Local weights
        float input_weights[1024];
        for(int m = 0 ; m < 1024 ;m++)
        {
            input_weights[m] = input1[((ff * 1024) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 1024; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            for (int yy = 0; yy < 7; ++yy)
            {
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
                temp_out[yy][xx] += input2[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}



//118

//P kernel
__kernel void P_block4_unit_1_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 41472; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
    }
}
//119

__kernel void  block4_unit_1_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    float l_input[9*9];
    for (int ff = 0; ff < 512; ++ff) {
        //Local weights
		
        float input_weights[3*3*512];
        for(int m = 0 ; m < 3*3*512 ; m++){
            input_weights[m] = input1[((ff * 3*3*512) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        
        for (int rc = 0; rc < 512; ++rc)
        {
            for (int i = 0; i < 9*9; i++){
                l_input[i] = input0[9*9*rc+i];
            }
            #pragma unroll 2
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
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input2[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}


//121

__kernel void  block4_unit_1_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1,__global float* restrict input2) {
    
    float l_input[49];
    for (int ff = 0; ff < 2048; ++ff) {
        
        //Local weights
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++)
        {
            input_weights[m] = input1[((ff * 512) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            for (int yy = 0; yy < 7; ++yy)
            {
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
                temp_out[yy][xx] += input2[ff];
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
                
            }
        }
    }
}


__kernel void  block4_unit_1_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
    }
}


__kernel void Mul1_1898_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f);
    }
}




__kernel void  block4_unit_2_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    float l_input[49];
    for (int ff = 0; ff < 512; ++ff) {
        
        //Local weights
        float input_weights[2048];
        for(int m = 0 ; m < 2048 ;m++){
            input_weights[m] = input1[((ff * 2048) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 2048; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            for (int yy = 0; yy < 7; ++yy)
            {
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
                temp_out[yy][xx] += input2[ff];
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
    for (int ff = 0; ff < 512; ++ff) {
        //Local weights
        float input_weights[3*3*512];
        for(int m = 0 ; m < 3*3*512 ; m++){
            input_weights[m] = input1[((ff * 3*3*512) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; ++rc)
        {
            for (int i = 0; i < 9*9; i++){
                l_input[i] = input0[9*9*rc+i];
            }
			#pragma unroll 2
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
        for (int yy = 0; yy < 7; ++yy)
        { 
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input2[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}



__kernel void  block4_unit_2_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    
    float l_input[49];
    for (int ff = 0; ff < 2048; ++ff) {
        //Local weights
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            for (int yy = 0; yy < 7; ++yy)
            {
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
                temp_out[yy][xx] += input2[ff];
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


__kernel void  Mul1_1925_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f);
    }
}

__kernel void  block4_unit_3_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    
    float l_input[49];
    for (int ff = 0; ff < 512; ++ff) {
        //Local weights
        float input_weights[2048];
        for(int m = 0 ; m < 2048 ;m++){
            input_weights[m] = input1[((ff * 2048) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 2048; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            for (int yy = 0; yy < 7; ++yy)
            {
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
                temp_out[yy][xx] += input2[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}


__kernel void P_block4_unit_3_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 41472; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
    }
}


__kernel void  block4_unit_3_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    
    float l_input[9*9];
    for (int ff = 0; ff < 512; ++ff) {
        
        //Local weights
        float input_weights[3*3*512];
        for(int m = 0 ; m < 3*3*512 ; m++){
            input_weights[m] = input1[((ff * 3*3*512) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        
        for (int rc = 0; rc < 512; ++rc)
        {
            for (int i = 0; i < 9*9; i++){
                l_input[i] = input0[9*9*rc+i];
            }
			#pragma unroll 2
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
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input2[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
                
                
            }
        }
    }
}


__kernel void  block4_unit_3_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    
    float l_input[49];
    for (int ff = 0; ff < 2048; ++ff) {
        //Local weights
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
        float temp_out[7][7];
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            for (int yy = 0; yy < 7; ++yy)
            {
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
                temp_out[yy][xx] += input2[ff];
                compute[((((ff * 7) + yy) * 7) + xx)] = temp_out[yy][xx];
            }
        }
    }
}



__kernel void block4_unit_3_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
    }
}


__kernel void  Mul1_1952_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f);
    }
}


__kernel void pool5(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 2048; ++ax1)
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


__kernel void  logits_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    
    __local  float input_bias[1001];
    for(int b = 0; b < 1001; b++){
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 1001; ++ff) {
        //Local weights
        float input_weights[2048];
        for(int m = 0 ; m < 2048 ;m++){
            input_weights[m] = input1[((ff * 2048) + m)];
        }
        float temp_0 = input_bias[ff];
        float temp_1 = 0.0;
        for (int rc = 0; rc < 2048; ++rc) {
            temp_1 += (input0[rc] * input_weights[(rc)]);
        }
        temp_0 += temp_1;
        temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
        compute[ff] = temp_0;
    }
}

//143 //144

__kernel void logits_Conv2D_Permute_(__global float* restrict tensor, __global float* restrict input0, __global float* restrict input1) {
    for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
        tensor[ax0_ax1_fused_inner] = (input0[ax0_ax1_fused_inner] + input1[ax0_ax1_fused_inner]);
    }
}
