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

