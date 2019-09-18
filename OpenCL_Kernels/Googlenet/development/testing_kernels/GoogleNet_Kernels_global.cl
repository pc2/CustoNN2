__kernel void Padding_Conv2d_1a_7x7_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 157323; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((458 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) < 51754)) && (2 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229) < 226)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) / 229) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52441)) + -1350)] : 0.000000e+00f);
    }
}

__kernel void Conv2d_1a_7x7_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[64];
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 64; ++ff)
    {
        //Local weights 
        float  local_weight[7*7*3];
        for(int m = 0 ; m < 7*7*3 ;m++){
            local_weight[m] = input1[((ff * 7*7*3) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[112][112];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 112; l++ ){
            for (int j = 0; j < 112; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 3; ++rc)
        {
            //Store 1 slice of input image
             float image_slice[229*229];
             #pragma unroll 32
            for (int in = 0; in < 229*229; in++){
                    image_slice[in]= input0[(229*229*rc)+in];
            }
            for (int yy = 0; yy < 112; ++yy)
            {
                #pragma unroll 4
                for (int xx = 0; xx < 112; ++xx)
                {
                    float temp_0 = 0;
                    float temp_2 = 0;
                        #pragma unroll
                        for (int ry = 0; ry < 7; ++ry)
                        {
                            float temp_1 = 0;
                            #pragma unroll
                            for (int rx = 0; rx < 7; ++rx)
                            {   
                                temp_1 +=  (image_slice[(yy * 458) + (ry * 229) + (xx * 2) + rx] * local_weight[(((((rc) * 7) + ry) * 7) + rx)]);
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
            for (int yy = 0; yy < 112; ++yy)
            {
                for (int xx = 0; xx < 112; ++xx)
                {
                    temp_out[yy][xx] += input_bias[ff];
                    //RELU
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f; 
                    compute[((((ff * 112) + yy) * 112) + xx)] = temp_out[yy][xx];
                }
            }
    }
}

__kernel void MaxPool_2a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
     

    for (int ax1 = 0; ax1 < 64; ++ax1)
    {
        //Store 1 slice of data
        float input0_l[112 * 112];
        //#pragma unroll 112
        for (int i = 0; i < 112 * 112; i++)
        {
            input0_l[i] = input0[(ax1*112*112) + i];
        }
        for (int ax2 = 0; ax2 < 56; ++ax2)
        {
            for (int ax3 = 0; ax3 < 56; ++ax3)
            {
                float tensor2 = -3.402823e+38f;
                //#pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
                    #pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor2 = max(tensor2, (float)((((ax2 * 2) < (112 - rv)) && ((ax3 * 2) < (112 - rv1))) ? input0_l[(((((((ax2) * 2) + rv) * 56) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
                tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = tensor2 ;
            }
        }
    }
}

__kernel void Conv2d_2b_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[64];
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 64; ++ff)
    {
        //Local weights 
        float input_weights[64];
        for(int m = 0 ; m < 64 ;m++){
            input_weights[m] = input1[((ff * 64) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[56][56];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 56; l++ ){
            for (int j = 0; j < 56; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 64; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[56*56];
            for (int in = 0; in < 56*56; in++){
                image_slice[in] = input0[(56*56*rc)+in];
            }
            //#pragma unroll 4
            for (int yy = 0; yy < 56; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 56; ++xx)
                {
                    temp_out[yy][xx] += (image_slice[(yy * 56) + xx] * input_weights[rc]);
                }
            }
        }
        //Summarize the results depthwise.
        #pragma loop_coalesce
        for (int yy = 0; yy < 56; ++yy)
        {
                for (int xx = 0; xx < 56; ++xx)
                {   
                    temp_out[yy][xx] += input_bias[ff];
                    //Relu
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 56) + yy) * 56) + xx)] =  temp_out[yy][xx];
                }
        }
    }
}

__kernel void Padding_Conv2d_2c_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f);
    }
}

__kernel void Conv2d_2c_3x3_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[192];
    for(int b = 0; b < 192; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 192; ++ff)
    {
        //Local weights 
        float local_weight[3*3*64];
        for(int m = 0 ; m < 3*3*64 ; m++){
            local_weight[m] = input1[((ff * 3*3*64) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[56][56];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 56; l++ ){
            for (int j = 0; j < 56; j++){
                temp_out[l][j] = 0.0;
            }
        }
        for (int rc = 0; rc < 64; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[58*58];
            #pragma unroll 29
            for (int in = 0; in < 58*58; in++){
                image_slice[in] = input0[(58*58*rc)+in];
            }
            //#pragma unroll 2
            for (int yy = 0; yy < 56; ++yy)
            {
                #pragma unroll 
                for (int xx = 0; xx < 56; ++xx)
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
                                temp_1 +=  (image_slice[((yy+ry) * 58) + (xx) + rx ] * local_weight[(((((rc) * 3) + ry) * 3) + rx)]);
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
            for (int yy = 0; yy < 56; ++yy)
            {
                for (int xx = 0; xx < 56; ++xx)
                {
                    temp_out[yy][xx] += input_bias[ff];
                    //RELU
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f; 
                   compute[((((ff * 56) + yy) * 56) + xx)] = temp_out[yy][xx];
                }
            }
    }
}

__kernel void MaxPool_3a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
  for (int ax1 = 0; ax1 < 192; ++ax1)
    {
        float inputl[56 * 56];
        for (int i = 0; i < 56 * 56; i++)
        {
            inputl[i] = input0[(ax1*56*56)+i];
        }
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {
                float tensor1 = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    #pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor1 = max(tensor1, (float)((((ax2 * 2) < (56 - rv)) && ((ax3 * 2) < (56 - rv1))) ? inputl[((((((( ax2) * 2) + rv) * 28) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
                tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = tensor1;
            }
        }
    }
}


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
__kernel void Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{

    __local  float l_bias[128];
    #pragma unroll
    for(int b = 0; b < 128; b++){
        l_bias[b] = input2[b];
    }


    for (int ff = 0; ff < 128; ++ff)
    {

        float l_weights[256];
        // #pragma unroll
        for(int m = 0 ; m < 256 ;m++){
            l_weights[m] = input1[((ff * 256) + m)];
        }

        float temp_out[28][28];
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }


        for (int rc = 0; rc < 256; ++rc)
        {

            float image_slice[28*28];
            // #pragma unroll 28
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }


            #pragma unroll 4
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += image_slice[(yy * 28) + xx] * l_weights[rc];
                }
            }
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += l_bias[ff];
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 28) + yy) * 28) + xx)] =  temp_out[yy][xx];
                }
        }

    }
}

__kernel void Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{


    __local  float l_bias[128];
    // #pragma unroll
    for(int b = 0; b < 128; b++){
        l_bias[b] = input2[b];
    }


    for (int ff = 0; ff < 128; ++ff)
    {

        float l_weights[256];
        // #pragma unroll
        for(int m = 0 ; m < 256 ;m++){
            l_weights[m] = input1[((ff * 256) + m)];
        }

        float temp_out[28][28];
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }


        for (int rc = 0; rc < 256; ++rc)
        {

            float image_slice[28*28];
            // #pragma unroll 28
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }

            #pragma unroll 4
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += image_slice[(yy * 28) + xx] * l_weights[rc];
                }
            }
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += l_bias[ff];
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 28) + yy) * 28) + xx)] =  temp_out[yy][xx];
                }
        }

    }
}

__kernel void Padding_Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
    }
}

__kernel void Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{

    __local  float l_bias[192];
    // #pragma unroll
    for(int b = 0; b < 192; b++){
        l_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 192; ++ff)
    {

        float l_weights[3*3*128];
        // #pragma unroll
        for(int m = 0 ; m < 3*3*128; m++){
            l_weights[m] = input1[((ff * 3*3*128) + m)];
        }

        float temp_out[28][28];
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }


        for (int rc = 0; rc < 128; ++rc)
        {

            //Store 1 slice of input image
            float image_slice[30*30];
            // #pragma unroll 28
            for (int in = 0; in < 30*30; in++){
                image_slice[in] = input0[(30*30*rc)+in];
            }

            #pragma unroll 4
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
                            temp_1 +=  (image_slice[((yy+ry) * 30) + (xx) + rx ] * l_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 +=temp_1;
                    }
                    temp_0 += temp_2;
                    temp_out[yy][xx] += temp_0;

                }
            }
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += l_bias[ff];
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 28) + yy) * 28) + xx)] =  temp_out[yy][xx];
                }
        }


    }   
}

__kernel void Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    __local  float l_bias[128];
    // #pragma unroll
    for(int b = 0; b < 128; b++){
        l_bias[b] = input2[b];
    }


    for (int ff = 0; ff < 32; ++ff)
    {

        float l_weights[256];
        // #pragma unroll
        for(int m = 0 ; m < 256 ;m++){
            l_weights[m] = input1[((ff * 256) + m)];
        }

        float temp_out[28][28];
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }


        for (int rc = 0; rc < 256; ++rc)
        {

            float image_slice[28*28];
            // #pragma unroll 28
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }

            #pragma unroll 4
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += image_slice[(yy * 28) + xx] * l_weights[rc];
                }
            }
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += l_bias[ff];
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 28) + yy) * 28) + xx)] =  temp_out[yy][xx];
                }
        }

    }
}

__kernel void Padding_Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28800; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
    }
}

__kernel void Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    __local  float l_bias[96];
    // #pragma unroll
    for(int b = 0; b < 96; b++){
        l_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 96; ++ff)
    {

        float l_weights[3*3*32];
        // #pragma unroll
        for(int m = 0 ; m < 3*3*32; m++){
            l_weights[m] = input1[((ff * 3*3*32) + m)];
        }

        float temp_out[28][28];
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }


        for (int rc = 0; rc < 32; ++rc)
        {

            //Store 1 slice of input image
            float image_slice[30*30];
            // #pragma unroll 28
            for (int in = 0; in < 30*30; in++){
                image_slice[in] = input0[(30*30*rc)+in];
            }

            #pragma unroll 2
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
                            temp_1 +=  (image_slice[((yy+ry) * 30) + (xx) + rx ] * l_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 +=temp_1;
                    }
                    temp_0 += temp_2;
                    temp_out[yy][xx] += temp_0;

                }
            }
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += l_bias[ff];
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 28) + yy) * 28) + xx)] =  temp_out[yy][xx];
                }
        }


    } 
}

__kernel void Mixed_3c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 256; ++ax1)
    {

        float input0_l[28 * 28];

        for (int i = 0; i < 28 * 28; i++)
        {
            input0_l[i] = input0[(ax1*28*28) + i];
        }


        #pragma loop_coalesce 2
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {

                float tensor1 = -3.402823e+38f;

                #pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
                    #pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor1 = max(tensor1, (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? input0_l[(((((ax2 + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
                    }
                }
                tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = tensor1;
            }
        }
    }
}

__kernel void Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{

    __local  float l_bias[128];
    // #pragma unroll
    for(int b = 0; b < 128; b++){
        l_bias[b] = input2[b];
    }


    for (int ff = 0; ff < 64; ++ff)
    {

        float l_weights[256];
        // #pragma unroll
        for(int m = 0 ; m < 256 ;m++){
            l_weights[m] = input1[((ff * 256) + m)];
        }

        float temp_out[28][28];
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }


        for (int rc = 0; rc < 256; ++rc)
        {

            float image_slice[28*28];
            #pragma unroll 28
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }

            #pragma unroll 4
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += image_slice[(yy * 28) + xx] * l_weights[rc];
                }
            }
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += l_bias[ff];
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    compute[((((ff * 28) + yy) * 28) + xx)] =  temp_out[yy][xx];
                }
        }

    }
}

__kernel void Mixed_3c_concat(__global float *restrict T_transpose, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 376320; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((326144 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -326144)] : (float)((250880 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -250880)] : (float)((100352 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -100352)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
__kernel void MaxPool_4a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 480; ++ax1)
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
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((ax2 * 2) < (28 - rv)) && ((ax3 * 2) < (28 - rv1))) ? input0[((((((((ax1 * 14) + ax2) * 2) + rv) * 14) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
													 {
//local memory for biases
    __local float input_bias[192];
//#pragma unroll 32
    for (int j = 0; j < 192; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 192; ++ff)
    {
        //local memory for weights
        float input_weight[480];
	//	#pragma unroll 64

        for (int k = 0; k < 480; k++){
            input_weight[k] = input1[((ff * 480) + k)];
        }
        
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 480; rc++)
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

__kernel void Mixed_4b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{

    //local memory for biases
    __local float input_bias[96];
	//#pragma unroll 32
    for (int j = 0; j < 96; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 96; ++ff)
    {
        //local memory for weights
        float input_weight[480];
//	#pragma unroll 64

        for (int k = 0; k < 480; k++){
            input_weight[k] = input1[((ff * 480) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 480; rc++)
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

__kernel void Padding_Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 24576; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}

__kernel void Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{

   //local memory for biases
    __local float input_bias[208];
   // #pragma unroll 32
    for (int j = 0; j < 208; j++){
        input_bias[j] = input2[j];
    }
    float l_input[16*16];
    for (int ff = 0; ff < 208; ++ff)
    {
        //local memory for weights
        float input_weight[3*3*96];
//#pragma unroll 128
        for (int k = 0; k < 3*3*96; k++){
            input_weight[k] = input1[((ff * 3*3*96) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 96; ++rc)
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

__kernel void Mixed_4b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{

    //local memory for biases
    __local float input_bias[16];
    //#pragma unroll
//#pragma unroll
    for (int j = 0; j < 16; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 16; ++ff)
    {
        //local memory for weights
        float input_weight[480];
    //    #pragma unroll 64
        for (int k = 0; k < 480; k++){
            input_weight[k] = input1[((ff * 480) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 480; rc++)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
            
//#pragma unroll
#pragma unroll 4
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 14; ++xx)
                {
                    temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
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
__kernel void Padding_Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 4096; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
   //local memory for biases
    __local float input_bias[48];
   // #pragma unroll 32
    for (int j = 0; j < 48; j++){
        input_bias[j] = input2[j];
    }
    float l_input[16*16];
    for (int ff = 0; ff < 48; ++ff)
    {
        //local memory for weights
        float input_weight[3*3*16];
    //   #pragma unroll 64
        for (int k = 0; k < 3*3*16; k++){
            input_weight[k] = input1[((ff * 3*3*16) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 16; ++rc)
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


__kernel void Mixed_4b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 480; ++ax1)
    {
        float input1[14*14];
        for (int i = 0; i < 14*14; i++)
        {
            input1[i] = input0[(ax1*14*14)+i];
        }
        for (int ax2 = 0; ax2 < 14; ++ax2)
        {
#pragma unroll
            for (int ax3 = 0; ax3 < 14; ++ax3)
            {
                tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
#pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
#pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input1[((((((ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_4b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{

    //local memory for biases
    __local float input_bias[64];
//#pragma unroll
    for (int j = 0; j < 64; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 64; ++ff)
    {
        //local memory for weights
        float input_weight[480];
//#pragma unroll 32
        for (int k = 0; k < 480; k++){
            input_weight[k] = input1[((ff * 480) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 480; rc++)
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
                    temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
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

__kernel void Mixed_4b_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
__kernel void Mixed_4c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //local memory for biases
    __local float input_bias[160];
//#pragma unroll 4
    for (int j = 0; j < 160; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 160; ++ff)
    {
        //local memory for weights
        float input_weight[512];
       // #pragma unroll 4
        for (int k = 0; k < 512; k++){
            input_weight[k] = input1[((ff * 512) + k)];
        }
        
        float temp_out[14][14];
        #pragma loop_coalesce
        for (int l = 0; l < 14; l++){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 512; ++rc)
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

__kernel void Mixed_4c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    
    //local memory for biases
    __local float input_bias[112];
    // #pragma unroll 32
    for (int j = 0; j < 112; j++){
        input_bias[j] = input2[j];
    }
    
    float l_input[196];
    for (int ff = 0; ff < 112; ++ff)
    {
        //local memory for weights
        float input_weight[512];
//#pragma unroll 32
        for (int k = 0; k < 512; k++){
            input_weight[k] = input1[((ff * 512) + k)];
        }
        float temp_out[14][14];
        #pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 512; ++rc)
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

__kernel void Padding_Mixed_4c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28672; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    
    //local memory for biases
    __local float input_bias[224];
//#pragma unroll 32
    for (int j = 0; j < 224; j++){
        input_bias[j] = input2[j];
    }
    float l_input[16*16];
    for (int ff = 0; ff < 224; ++ff)
    {
        //local memory for weights
        float input_weight[3*3*112];
        //#pragma unroll 32
        for (int k = 0; k < 3*3*112; k++){
            input_weight[k] = input1[((ff * 3*3*112) + k)];
        }
        float temp_out[14][14];
        #pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 112; ++rc)
        {
            #pragma unroll 16
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

__kernel void Mixed_4c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    
    //local memory for biases
    __local float input_bias[24];
//#pragma unroll
    for (int j = 0; j < 24; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 24; ++ff)
    {
        //local memory for weights
        float input_weight[512];
//#pragma unroll 32
        for (int k = 0; k < 512; k++){
            input_weight[k] = input1[((ff * 512) + k)];
        }
        float temp_out[14][14];
        #pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        
        for (int rc = 0; rc < 512; ++rc)
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

__kernel void Padding_Mixed_4c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    
    //local memory for biases
    __local float input_bias[64];
//#pragma unroll 32
    for (int j = 0; j < 64; j++){
        input_bias[j] = input2[j];
    }
    float l_input[16*16];
    for (int ff = 0; ff < 64; ++ff)
    {
        //local memory for weights
        float input_weight[3*3*24];
//#pragma unroll 32
        for (int k = 0; k < 3*3*24; k++){
            input_weight[k] = input1[((ff * 3*3*24) + k)];
        }
        float temp_out[14][14];
        #pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 24; ++rc)
        {
            
            #pragma unroll 16
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

__kernel void Mixed_4c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 512; ++ax1)
    {
        float input1[14*14];
        for (int i = 0; i < 14*14; i++)
        {
            input1[i] = input0[(ax1*14*14)+i];
        }
        for (int ax2 = 0; ax2 < 14; ++ax2)
        {
#pragma unroll
            for (int ax3 = 0; ax3 < 14; ++ax3)
            {
                tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
#pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
#pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input1[((((((ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_4c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     constant float *restrict input2)
{
    //local memory for biases
    __local float input_bias[64];
//#pragma unroll 32
    for (int j = 0; j < 64; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 64; ++ff)
    {
        //local memory for weights
        float input_weight[512];
//#pragma unroll 32
        for (int k = 0; k < 512; k++){
            input_weight[k] = input1[((ff * 512) + k)];
        }
        float temp_out[14][14];
        #pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
//#pragma unroll 4
        for (int rc = 0; rc < 512; ++rc)
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
__kernel void Mixed_4c_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((75264 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -75264)] : (float)((31360 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -31360)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
__kernel void Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
   //Local memory for Biases:
    __local  float input_bias[128];
	#pragma unroll 8
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }


    float l_input[196];
	for (int ff = 0; ff < 128; ++ff)
    {
        float input_weights[512];
		#pragma unroll 8
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		
		float temp_out[14][14];
		#pragma loop_coalesce
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

__kernel void Mixed_4d_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
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
         //Local weights 
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		float temp_out[14][14];
		#pragma loop_coalesce
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
__kernel void Padding_Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 32768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
   	//Local memory for Biases:
    __local  float input_bias[256];
    for(int b = 0; b < 256; b++){
        input_bias[b] = input2[b];
    }


    float l_input[16*16];
    for (int ff = 0; ff < 256; ++ff)
    {
        //Local weights 
        float input_weights[3*3*128];
        for(int m = 0 ; m < 3*3*128 ; m++){
            input_weights[m] = input1[((ff * 3*3*128) + m)];
        }
		float temp_out[14][14];
		#pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
		
		for (int rc = 0; rc < 128; ++rc)
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
					temp_out[yy][xx] +=temp_1;
					
						
					float temp_2 =0.0;
                    #pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_2 += l_input[(yy+2) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
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

__kernel void Mixed_4d_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[24];
    for(int b = 0; b < 24; b++){
        input_bias[b] = input2[b];
    }

  
	float l_input[196];
    for (int ff = 0; ff < 24; ++ff)
    {
        //Local weights 
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		
		float temp_out[14][14];
		#pragma loop_coalesce
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
__kernel void Padding_Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[64];
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }

	float l_input[16*16];
    for (int ff = 0; ff < 64; ++ff)
    {
         //Local weights 
        float input_weights[3*3*24];
        for(int m = 0 ; m < 3*3*24 ; m++){
            input_weights[m] = input1[((ff * 3*3*24) + m)];
        }
		
		float temp_out[14][14];
		#pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
		for (int rc = 0; rc < 24; ++rc)
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
					float temp_0 = 0.0;
					#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_0 += l_input[(yy+0) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
                    }
					temp_out[yy][xx] += temp_0;
					
					float temp_1= 0.0;
					#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_1 += l_input[(yy+1) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
                    }
					temp_out[yy][xx] +=temp_1;
					
					float temp_2 = 0.0;
					#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_2 += l_input[(yy+2) * 16 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
                    }
					temp_out[yy][xx] +=temp_2;
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

__kernel void Mixed_4d_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 512; ++ax1)
    {
		float input1[14*14];
        for (int i = 0; i < 14 * 14; i++)
        {
            input1[i] = input0[(ax1*14*14)+i];
        }
        for (int ax2 = 0; ax2 < 14; ++ax2)
        {
			#pragma unroll
            for (int ax3 = 0; ax3 < 14; ++ax3)
            {
                tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
				#pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
					#pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input1[((((((ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_4d_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
   //Local memory for Biases:
    __local  float input_bias[64];
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }

	float l_input[196];
    for (int ff = 0; ff < 64; ++ff)
    {
        //Local weights 
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		
		float temp_out[14][14];
		#pragma loop_coalesce
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

__kernel void Mixed_4d_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((75264 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -75264)] : (float)((25088 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -25088)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
__kernel void Mixed_4e_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{

     //Local memory for Biases:
    __local  float input_bias[112];
    for(int b = 0; b < 112; b++){
        input_bias[b] = input2[b];
    }

	float l_input[196];
    for (int ff = 0; ff < 112; ++ff)
    {
		//Local weights 
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		float temp_out[14][14];
		#pragma loop_coalesce
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

__kernel void Mixed_4e_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
   //Local memory for Biases:
    __local  float input_bias[144];
    for(int b = 0; b < 144; b++){
        input_bias[b] = input2[b];
    }

	float l_input[196];
    for (int ff = 0; ff < 144; ++ff)
    {
	 //Local weights 
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
		}
		float temp_out[14][14];
		#pragma loop_coalesce
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
    for(int b = 0; b < 288; b++){
        input_bias[b] = input2[b];
    }

	float l_input[16*16];
    for (int ff = 0; ff < 288; ++ff)
    {
	 //Local weights 
        float input_weights[3*3*144];
        for(int m = 0 ; m < 3*3*144 ; m++){
            input_weights[m] = input1[((ff * 3*3*144) + m)];
        }
		float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
		for (int rc = 0; rc < 144; ++rc)
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

__kernel void Mixed_4e_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
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
	//Local weights 
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		
		float temp_out[14][14];
		#pragma loop_coalesce
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
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }

	
	float l_input[16*16];
    for (int ff = 0; ff < 64; ++ff)
    {
		//Local weights 
        float input_weights[3*3*32];
        for(int m = 0 ; m < 3*3*32 ; m++){
            input_weights[m] = input1[((ff * 3*3*32) + m)];
        }
		float temp_out[14][14];
		#pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
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

__kernel void Mixed_4e_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 512; ++ax1)
    {
		float input1[14*14];
        for (int i = 0; i < 14 * 14; i++)
        {
            input1[i] = input0[(ax1*14*14)+i];
        }
        for (int ax2 = 0; ax2 < 14; ++ax2)
        {
			#pragma unroll
            for (int ax3 = 0; ax3 < 14; ++ax3)
            {
                tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
				#pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
					#pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input1[((((((ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
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
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }
	float l_input[196];
    for (int ff = 0; ff < 64; ++ff)
    {
		//Local weights 
        float input_weights[512];
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		float temp_out[14][14];
		#pragma loop_coalesce
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

__kernel void Mixed_4e_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((90944 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -90944)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((21952 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -21952)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
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
#pragma unroll
        for (int yy = 0; yy < 14; ++yy)
        {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
            }
            
        }
        }
        for (int yy = 0; yy < 14; ++yy)
        {
			#pragma unroll
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
__kernel void MaxPool_5a_2x2_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 832; ++ax1)
    {
        for (int ax2 = 0; ax2 < 7; ++ax2)
        {
            for (int ax3 = 0; ax3 < 7; ++ax3)
            {
                tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 2; ++rv)
                {
                    for (int rv1 = 0; rv1 < 2; ++rv1)
                    {
                        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], input0[((((((((ax1 * 7) + ax2) * 2) + rv) * 7) + ax3) * 2) + rv1)]);
                    }
                }
            }
        }
    }
}

__kernel void Mixed_5b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[256];
    for(int b = 0; b < 256; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[49];
    for (int ff = 0; ff < 256; ++ff)
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
                temp_out[l][j] = 0;
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

__kernel void Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[160];
    for(int b = 0; b < 160; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[49];
    for (int ff = 0; ff < 160; ++ff)
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
                temp_out[l][j] = 0;
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
__kernel void Padding_Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 12960; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
    }
}
__kernel void Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    __local  float input_bias[320];
    for(int b = 0; b < 320; b++){
        input_bias[b] = input2[b];
    }
    float l_input[9*9];
    for (int ff = 0; ff < 320; ++ff)
    {
        float input_weights[3*3*160];
        for (int k = 0; k < 3*3*160; k++){
            input_weights[k] = input1[((ff *3*3*160) + k)];
        }
        float temp_out[7][7];
#pragma loop_coalesce
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0.0;
            }
        }
        for (int rc = 0; rc < 160; ++rc)
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

__kernel void Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Local memory for Biases:
    __local  float input_bias[32];
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
    }
    
    float l_input[49];
    for (int ff = 0; ff < 32; ++ff)
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
                temp_out[l][j] = 0;
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
__kernel void Padding_Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 2592; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
    }
}
__kernel void Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    __local  float input_bias[128];
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }
    float l_input[9*9];
    for (int ff = 0; ff < 128; ++ff)
    {
        float input_weights[3*3*32];
        for (int k = 0; k < 3*3*32; k++){
            input_weights[k] = input1[((ff *3*3*32) + k)];
        }
        float temp_out[7][7];
#pragma loop_coalesce
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0.0;
            }
        }
#pragma unroll 2
        for (int rc = 0; rc < 32; ++rc)
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
        //
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


__kernel void Mixed_5b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
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

__kernel void Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
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
                temp_out[l][j] = 0;
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

__kernel void Mixed_5b_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((34496 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -34496)] : (float)((28224 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -28224)] : (float)((12544 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -12544)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
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