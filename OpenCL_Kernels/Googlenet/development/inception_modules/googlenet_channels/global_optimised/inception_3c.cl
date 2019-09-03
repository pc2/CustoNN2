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