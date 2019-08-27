__kernel void Mixed_4e_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{

     //Local memory for Biases:
    __local  float input_bias[112];
	#pragma unroll 64
    for(int b = 0; b < 112; b++){
        input_bias[b] = input2[b];
    }

	float l_input[196];
    for (int ff = 0; ff < 112; ++ff)
    {
		//Local weights 
        float input_weights[512];
        #pragma unroll 64
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
			}
		}
		for (int rc = 0; rc < 512; rc++)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
			#pragma unroll 4
			for (int yy = 0; yy < 14; ++yy)
			{
				#pragma unroll
				for (int xx = 0; xx < 14; ++xx)
				{
                    temp_out[yy][xx] += (l_input[ yy * 14 + xx] * input_weights[rc]);
                }
			}
		}
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

__kernel void Mixed_4e_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
   //Local memory for Biases:
    __local  float input_bias[144];
	#pragma unroll 64
    for(int b = 0; b < 144; b++){
        input_bias[b] = input2[b];
    }

	float l_input[196];
    for (int ff = 0; ff < 144; ++ff)
    {
	 //Local weights 
        float input_weights[512];
        #pragma unroll 64
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
		}
		float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
			}
		}

		for (int rc = 0; rc < 512; rc++)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }	
			#pragma unroll 4
			for (int yy = 0; yy < 14; ++yy)
			{
				#pragma unroll
				for (int xx = 0; xx < 14; ++xx)
				{
                    temp_out[yy][xx] += (l_input[ yy * 14 + xx] * input_weights[(rc)]);
                }
			}
		}
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
__kernel void Padding_Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 36864; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
   //Local memory for Biases:
    __local  float input_bias[288];
	#pragma unroll 128
    for(int b = 0; b < 288; b++){
        input_bias[b] = input2[b];
    }

	float l_input[16*16];
    for (int ff = 0; ff < 288; ++ff)
    {
	 //Local weights 
        float input_weights[3*3*144];
        #pragma unroll 64
        for(int m = 0 ; m < 3*3*144 ; m++){
            input_weights[m] = input1[((ff * 3*3*144) + m)];
        }
		float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
		#pragma unroll 4
		for (int rc = 0; rc < 144; ++rc)
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
				float temp_0 = 0.0;
				#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_0 += l_input[(yy+0) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
                    }
					temp_out[yy][xx] += temp_0;
					
					float temp_1 = 0.0;
					#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_1 += l_input[(yy+1) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
                    }
					temp_out[yy][xx] += temp_1;
					
					
					float temp_2 = 0.0;
                    #pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_2 += l_input[(yy+2) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
                    }
					temp_out[yy][xx] += temp_2;
				}
			}
		}
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

__kernel void Mixed_4e_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[32];
	#pragma unroll
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
	}

	float l_input[196];
    for (int ff = 0; ff < 32; ++ff)
    {
	//Local weights 
        float input_weights[512];
        #pragma unroll 64
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		
		float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
		
        for (int rc = 0; rc < 512; rc++)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
			#pragma unroll 4
			for (int yy = 0; yy < 14; ++yy)
			{
				#pragma unroll
				for (int xx = 0; xx < 14; ++xx)
				{
                temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weights[rc]);
                }
			}
		}
		
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
__kernel void Padding_Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
   //Local memory for Biases:
    __local  float input_bias[64]; 
	#pragma unroll 
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }

	
	float l_input[16*16];
    for (int ff = 0; ff < 64; ++ff)
    {
		//Local weights 
        float input_weights[3*3*32];
        #pragma unroll 64
        for(int m = 0 ; m < 3*3*32 ; m++){
            input_weights[m] = input1[((ff * 3*3*32) + m)];
        }
		
		float temp_out[14][14];
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
			#pragma unroll 4
			for (int yy = 0; yy < 14; ++yy)
			{
				#pragma unroll
				for (int xx = 0; xx < 14; ++xx)
				{
					float temp_0 = 0;
					#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_0 += l_input[(yy+0) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
                    }
					temp_out[yy][xx] += temp_0;
					
					
					float temp_1 = 0;
					#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_1 += l_input[(yy+1) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
                    }
					temp_out[yy][xx] += temp_1;
					
					
					float temp_2 = 0;
					#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_2 += l_input[(yy+2) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
                    }
					temp_out[yy][xx] += temp_2;
                }
            }
        }
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

__kernel void Mixed_4e_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 512; ++ax1)
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

__kernel void Mixed_4e_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    	//Local memory for Biases:
    __local  float input_bias[64];
	#pragma unroll 
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }

	
	float l_input[196];
    for (int ff = 0; ff < 64; ++ff)
    {
		//Local weights 
        float input_weights[512];
        #pragma unroll 64
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
		for (int rc = 0; rc < 512; rc++)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
			#pragma unroll 4
			for (int yy = 0; yy < 14; ++yy)
			{
				#pragma unroll
				for (int xx = 0; xx < 14; ++xx)
				{
                 
                    temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weights[rc]);
                }
				 
			
            }
        }
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

__kernel void Mixed_4e_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((90944 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -90944)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((21952 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -21952)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
