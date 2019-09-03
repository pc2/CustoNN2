__kernel void Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
       //Local memory for Biases:
    __local  float input_bias[64];
    //#pragma unroll
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 64; ++ff)
    {
        //Local weights 
        float input_weights[192];
        //#pragma unroll 128
        for(int m = 0 ; m < 192 ;m++){
            input_weights[m] = input1[((ff * 192) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 192; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[28*28];
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }
            //#pragma unroll 2
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += (image_slice[(yy * 28) + xx] * input_weights[rc]);
                }

            }
        }
            //Summarize the results depthwise.
            #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += input_bias[ff];
                    //Relu
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                   compute[((((ff * 28) + yy) * 28) + xx)] = temp_out[yy][xx];
                }
            }
    }
}

__kernel void Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
         //Local memory for Biases:
    __local  float input_bias[96];
    //#pragma unroll 32
    for(int b = 0; b < 96; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 96; ++ff)
    {
        //Local weights 
        float input_weights[192];
        //#pragma unroll 64
        for(int m = 0 ; m < 192 ;m++){
            input_weights[m] = input1[((ff * 192) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 192; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[28*28];
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }
            //#pragma unroll 2
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += (image_slice[(yy * 28) + xx] * input_weights[rc]);
                }

            }
        }
        //Summarize the results depthwise.
        #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += input_bias[ff];
                    //Relu
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 28) + yy) * 28) + xx)] = temp_out[yy][xx];
                }
            }

    }
}

__kernel void Padding_Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 86400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
    }
}

__kernel void Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
        //Local memory for Biases:
    __local  float input_bias[128];
    //#pragma unroll 32
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 128; ++ff)
    {
        //Local weights 
        float local_weight[3*3*96];
        //#pragma unroll 32
        for(int m = 0 ; m < 3*3*96 ; m++){
            local_weight[m] = input1[((ff * 3*3*96) + m)];
        }

        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0.0;
            }
        }

        for (int rc = 0; rc < 96; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[30*30];
            #pragma unroll 30
            for (int in = 0; in < 30*30; in++){
                image_slice[in] = input0[(30*30*rc)+in];
            }
             //Convultion 3*3
             
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll 
                for (int xx = 0; xx < 28; ++xx)
                {
                        float temp_0 = 0;
                       
                        float temp_2 = 0;
                        #pragma unroll
                        for (int ry = 0; ry < 3; ++ry)
                        {
                            float temp_1 = 0;
                            #pragma unroll
                            for (int rx = 0; rx < 3; ++rx)
                            {
                                temp_1 +=  (image_slice[((yy+ry) * 30) + (xx) + rx ] * local_weight[(((((rc) * 3) + ry) * 3) + rx)]);
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
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += input_bias[ff];
                    //RELU
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f; 
                     compute[((((ff * 28) + yy) * 28) + xx)]  = temp_out[yy][xx];
                }
            }

    }
}

__kernel void Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
        //Local memory for Biases:
    __local  float input_bias[16];
    //#pragma unroll
    for(int b = 0; b < 16; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 16; ++ff)
    {
        //Local weights 
        float input_weights[192];
        //#pragma unroll 64
        for(int m = 0 ; m < 192 ;m++){
            input_weights[m] = input1[((ff * 192) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 192; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[28*28];
            //#pragma unroll 28
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }
            //#pragma unroll 2
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += (image_slice[yy * 28 + xx] * input_weights[rc]);
                }
            }
        }
            //Summarize the results depthwise.
            #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += input_bias[ff];
                    //Relu
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 28) + yy) * 28) + xx)] = temp_out[yy][xx];
                }
            }

        
    }
}

__kernel void Padding_Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 14400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
    }
}

__kernel void Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
        //Local memory for Biases:
    __local  float input_bias[32];
    //#pragma unroll
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 32; ++ff)
    {
        //Local weights 
        float local_weight[3*3*16];
        //#pragma unroll 64
        for(int m = 0 ; m < 3*3*16 ; m++){
            local_weight[m] = input1[((ff * 3*3*16) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0.0;
            }
        }
        for (int rc = 0; rc < 16; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[30*30];
            #pragma unroll 30
            for (int in = 0; in < 30*30; in++){
                image_slice[in] = input0[(30*30*rc)+in];
            }
            
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    float temp_0 = 0;
                        //Convultion 3*3
                        float temp_2 = 0;
                        #pragma unroll
                        for (int ry = 0; ry < 3; ++ry)
                        {
                            float temp_1 = 0;
                            #pragma unroll
                            for (int rx = 0; rx < 3; ++rx)
                            {
                                temp_1 +=  (image_slice[((yy+ry) * 30) + (xx) + rx ] * local_weight[(((((rc) * 3) + ry) * 3) + rx)]);
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
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += input_bias[ff];
                    //RELU
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f; 
                    compute[((((ff * 28) + yy) * 28) + xx)] =  temp_out[yy][xx];
                }
            }
    }
}


__kernel void Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 192; ++ax1)
    {
        float input1[28 * 28];
        for (int i = 0; i < 28 * 28; i++)
        {
            input1[i] = input0[(ax1*28*28)+i];
        }
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            //#pragma unroll
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {
                tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
                #pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
                    #pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? input1[((((((ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}
__kernel void Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
   //Local memory for Biases:
    __local  float input_bias[32];
    //#pragma unroll
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
    }


    for (int ff = 0; ff < 32; ++ff)
    {
        //Local weights 
        float input_weights[192];
        //#pragma unroll 64
        for(int m = 0 ; m < 192 ;m++){
            input_weights[m] = input1[((ff * 192) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 192; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[28*28];
            //#pragma unroll 28
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }
            #pragma unroll 2
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += (image_slice[yy * 28 + xx] * input_weights[rc]);
                }
            }
        }
            //Summarize the results depthwise.
            #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += input_bias[ff];
                    //Relu
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 28) + yy) * 28) + xx)] = temp_out[yy][xx];
                }
            }
    }
}

__kernel void Mixed_3b_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((175616 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -175616)] : (float)((150528 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -150528)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
