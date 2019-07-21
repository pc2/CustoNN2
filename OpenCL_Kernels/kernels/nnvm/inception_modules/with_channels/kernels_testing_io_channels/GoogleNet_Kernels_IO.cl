/**
 * 1st Inception module - Conv 1a to Mixed_3b_concat
 */

//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_3b_buffer
{
    float concat_3b_out_buffer[8];
} concat_3b_struct;
// IO Channels for inception 3b to 3c
channel concat_3b_struct concat_3b_out_channel __attribute__((depth(10))) __attribute__((io("concat_3b"))); // Channel Tx

__kernel void Padding_Conv2d_1a_7x7_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 157323; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((458 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) < 51754)) && (2 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229) < 226)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) / 229) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52441)) + -1350)] : 0.000000e+00f);
    }
}

__kernel void Conv2d_1a_7x7_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 112; ++yy)
        {
            for (int xx = 0; xx < 112; ++xx)
            {
                compute[((((ff * 112) + yy) * 112) + xx)] = input2[ff];
                for (int rc = 0; rc < 3; ++rc)
                {
                    for (int ry = 0; ry < 7; ++ry)
                    {
                        for (int rx = 0; rx < 7; ++rx)
                        {
                            compute[((((ff * 112) + yy) * 112) + xx)] = (compute[((((ff * 112) + yy) * 112) + xx)] + (input0[(((((rc * 52441) + (yy * 458)) + (ry * 229)) + (xx * 2)) + rx)] * input1[((((((ff * 3) + rc) * 7) + ry) * 7) + rx)]));
                        }
                    }
                }
                compute[((((ff * 112) + yy) * 112) + xx)] = (compute[((((ff * 112) + yy) * 112) + xx)] > 0) ? compute[((((ff * 112) + yy) * 112) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void MaxPool_2a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 64; ++ax1)
    {
        for (int ax2 = 0; ax2 < 56; ++ax2)
        {
            for (int ax3 = 0; ax3 < 56; ++ax3)
            {
                tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = max(tensor[((((ax1 * 56) + ax2) * 56) + ax3)], (float)((((ax2 * 2) < (112 - rv)) && ((ax3 * 2) < (112 - rv1))) ? input0[((((((((ax1 * 56) + ax2) * 2) + rv) * 56) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Conv2d_2b_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 56; ++yy)
        {
            for (int xx = 0; xx < 56; ++xx)
            {
                compute[((((ff * 56) + yy) * 56) + xx)] = input2[ff];
                for (int rc = 0; rc < 64; ++rc)
                {
                    compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 64) + rc)]));
                }
                compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] > 0) ? compute[((((ff * 56) + yy) * 56) + xx)] : 0.000000e+00f;
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
    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 56; ++yy)
        {
            for (int xx = 0; xx < 56; ++xx)
            {
                compute[((((ff * 56) + yy) * 56) + xx)] = input2[ff];
                for (int rc = 0; rc < 64; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (input0[((((((rc * 58) + yy) + ry) * 58) + xx) + rx)] * input1[((((((ff * 64) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] > 0) ? compute[((((ff * 56) + yy) * 56) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void MaxPool_3a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 192; ++ax1)
    {
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {
                tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((ax2 * 2) < (56 - rv)) && ((ax3 * 2) < (56 - rv1))) ? input0[((((((((ax1 * 28) + ax2) * 2) + rv) * 28) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 96; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
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
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 96; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 96) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 16; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
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
    for (int ff = 0; ff < 32; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 16; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 16) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 192; ++ax1)
    {
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {
                tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? input0[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}
__kernel void Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{

    for (int ff = 0; ff < 32; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Mixed_3b_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    printf("Inside 3b concat\n");
    //struct to store 256 bits of data
    struct concat_3b_buffer out;
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        //T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((175616 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -175616)] : (float)((150528 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -150528)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        float result = (float)((175616 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -175616)] : (float)((150528 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -150528)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_3b_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_3b_out_channel, out);
        }
    }
} /**
 * 2nd Inception module - Inception 3c
 */
//Enable the channel extension

typedef struct concat_3c_buffer
{
    float concat_3c_out_buffer[8];
} concat_3c_struct;

// IO Channels for inception 3b to 3c
channel concat_3b_struct concat_3c_in_channel __attribute__((depth(10))) __attribute__((io("concat_3b")));  // Channel Rx
channel concat_3c_struct concat_3c_out_channel __attribute__((depth(10))) __attribute__((io("concat_3c"))); // Channel Tx

channel concat_3b_struct concat_3c_in_b0_channel __attribute__((depth(10))); // internal channel Branch 1
channel concat_3b_struct concat_3c_in_b1_channel __attribute__((depth(10))); // internal channel Branch 2
channel concat_3b_struct concat_3c_in_b2_channel __attribute__((depth(10))); // internal channel Branch 3
channel concat_3b_struct concat_3c_in_b3_channel __attribute__((depth(10))); // internal channel Branch 4

//Auto run kernel to feed 3b results to 4 branches
__kernel void feeder_3c()
{
    printf("Inside 3c feeder\n");
    for (int i = 0; i < 25088; i++)
    {
        struct concat_3b_buffer input = read_channel_intel(concat_3c_in_channel);
        write_channel_intel(concat_3c_in_b0_channel, input);
        write_channel_intel(concat_3c_in_b1_channel, input);
        write_channel_intel(concat_3c_in_b2_channel, input);
        write_channel_intel(concat_3c_in_b3_channel, input);
    }
}

__kernel void Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    printf("Inside 3c b0 1*1\n");
    //Read Input from IO channel
    float convInput[200704];
    // 200704/8 = 25088
    for (int i = 0; i < 25088; i++)
    {
        //struct to store 256 bits of data
        struct concat_3b_buffer in;
        in = read_channel_intel(concat_3c_in_b0_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_3b_out_buffer[k];
        }
    }

    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 256; ++rc)
                {
                    compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (convInput[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? (compute[((((ff * 28) + yy) * 28) + xx)]) : 0.000000e+00f;
            }
        }
    }
}

__kernel void Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    printf("Inside Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D\n");
    //Read Input from IO channel
    float convInput[200704];
    for (int i = 0; i < 25088; i++)
    {
        //struct to store 256 bits of data
        struct concat_3b_buffer in;
        in = read_channel_intel(concat_3c_in_b1_channel);
//printf("Reading data from IO - %d\n",i);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_3b_out_buffer[k];
        }
    }
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 256; ++rc)
                {
                    compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (convInput[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
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
    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 128; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    printf("Inside Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D\n");
    //Read Input from IO channel
    float convInput[200704];
    for (int i = 0; i < 25088; i++)
    {
        //struct to store 256 bits of data
        struct concat_3b_buffer in;
        in = read_channel_intel(concat_3c_in_b2_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_3b_out_buffer[k];
        }
    }

    for (int ff = 0; ff < 32; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 256; ++rc)
                {
                    compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (convInput[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? +compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
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
    for (int ff = 0; ff < 96; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
                for (int rc = 0; rc < 32; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_3c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    printf("Inside Mixed_3c_Branch_3_MaxPool_0a_3x3_MaxPool\n");
    //Read Input from IO channel
    float maxInput[200704];
    for (int i = 0; i < 25088; i++)
    {
        //struct to store 256 bits of data
        struct concat_3b_buffer in;
        in = read_channel_intel(concat_3c_in_b3_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            maxInput[(i * 8) + k] = in.concat_3b_out_buffer[k];
        }
    }
    for (int ax1 = 0; ax1 < 256; ++ax1)
    {
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {
                tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? maxInput[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{

    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)

        {
            for (int xx = 0; xx < 28; ++xx)

            {

                compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];

                for (int rc = 0; rc < 256; ++rc)
                {
                    compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
                }
                compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? +compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Mixed_3c_concat(__global float *restrict T_transpose, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    //struct to store 256 bits of data
    struct concat_3c_buffer out;
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 376320; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        //T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((326144 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -326144)] : (float)((250880 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -250880)] : (float)((100352 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -100352)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        float result = (float)((326144 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -326144)] : (float)((250880 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -250880)] : (float)((100352 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -100352)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_3c_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_3c_out_channel, out);
        }
    }
} /**
 * 3rd Inception module - 4a to mixed_4b_concat
 */
//Enable the channel extension

typedef struct concat_4b_buffer
{
    float concat_4b_out_buffer[8];
} concat_4b_struct;

// IO Channels for inception 3c to 4a
channel concat_3c_struct concat_4a_in_channel __attribute__((depth(10))) __attribute__((io("concat_3c")));  // Channel Rx
channel concat_4b_struct concat_4b_out_channel __attribute__((depth(10))) __attribute__((io("concat_4b"))); // Channel Tx

channel concat_3c_struct concat_4a_in_max_channel __attribute__((depth(10))); // internal channel maxpool

//Auto run kernel to feed 3b results to 4 branches

__kernel void feeder_4a()
{
    for (int i = 0; i < 47040; i++)
    {
        struct concat_3c_buffer input = read_channel_intel(concat_4a_in_channel);
        write_channel_intel(concat_4a_in_max_channel, input);
    }
}

__kernel void MaxPool_4a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    //Read Input from IO channel
    float maxInput[376320];
    // 376320/8 = 47040
    for (int i = 0; i < 47040; i++)
    {
        //struct to store 256 bits of data
        struct concat_3c_buffer in;
        in = read_channel_intel(concat_4a_in_max_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            maxInput[(i * 8) + k] = in.concat_3c_out_buffer[k];
            //printf("\t 4a %d --- %f \n",(i*8)+k,maxInput[(i*8)+k]);
        }
    }
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
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((ax2 * 2) < (28 - rv)) && ((ax3 * 2) < (28 - rv1))) ? maxInput[((((((((ax1 * 14) + ax2) * 2) + rv) * 14) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
                //                if(ax3==0)
                //printf("\t 4a out %d --- %f \n", ((((ax1 * 14) + ax2) * 14) + ax3), tensor[((((ax1 * 14) + ax2) * 14) + ax3)]);
            }
        }
    }
}

__kernel void Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //printf("\t 4b input b0 60970 - 44814 154 ------ %f  %f %f \n", input0[60970], input0[44814], input0[154]);
    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 480; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? +compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
                //printf("\t 4b  b0 %d --- %f \n",((((ff * 14) + yy) * 14) + xx),compute[((((ff * 14) + yy) * 14) + xx)]);
            }
        }
    }
}

__kernel void Mixed_4b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    printf("\t 4b input b1 60970 - 44814 154 ------ %f  %f %f \n", input0[60970], input0[44814], input0[154]);
    for (int ff = 0; ff < 96; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 480; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? +compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
                //printf("\t 4b  b1 0a %d --- %f \n",((((ff * 14) + yy) * 14) + xx),compute[((((ff * 14) + yy) * 14) + xx)]);
            }
        }
    }
}

__kernel void Padding_Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 24576; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
        //printf("\t 4b  b1 0b Pad  %d --- %f \n",ax0_ax1_fused_ax2_fused_ax3_fused_inner,T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner]);
    }
}

__kernel void Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{

    for (int ff = 0; ff < 208; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 96; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 96) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.0;
                // printf("\t 4b  b1 0b %d --- %f \n",((((ff * 14) + yy) * 14) + xx),compute[((((ff * 14) + yy) * 14) + xx)]);
            }
        }
    }
}

__kernel void Mixed_4b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    printf("\t 4b input b2 60970 - 44814 154 ------ %f  %f %f \n", input0[60970], input0[44814], input0[154]);
    for (int ff = 0; ff < 16; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 480; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? +compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
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
    for (int ff = 0; ff < 48; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 16; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 16) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.0;
                //printf("\t 4b  b2 %d --- %f \n",((((ff * 14) + yy) * 14) + xx),compute[((((ff * 14) + yy) * 14) + xx)]);
            }
        }
    }
}

__kernel void Mixed_4b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    printf("\t 4b input b3 60970 - 44814 154 ------ %f  %f %f \n", input0[60970], input0[44814], input0[154]);
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
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input0[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
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

    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 480; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? +compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
                //printf("\t 4b  b3 %d --- %f \n",((((ff * 14) + yy) * 14) + xx),compute[((((ff * 14) + yy) * 14) + xx)]);
            }
        }
    }
}

__kernel void Mixed_4b_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    //struct to store 256 bits of data
    struct concat_4b_buffer out;
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        // T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        float result = T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        //printf("\t 4b %d - %d --- %f \n",ax0_ax1_fused_ax2_fused_ax3_fused_inner,ax0_ax1_fused_ax2_fused_ax3_fused_inner%8,result);
        out.concat_4b_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_4b_out_channel, out);
        }
    }
}
/**
 * 4th Inception module - Inception 4c
 */
//Enable the channel extension

typedef struct concat_4c_buffer
{
    float concat_4c_out_buffer[8];
} concat_4c_struct;

// IO Channels for inception 4b to 4c
channel concat_4b_struct concat_4c_in_channel __attribute__((depth(10))) __attribute__((io("concat_4b")));  // Channel Rx
channel concat_4c_struct concat_4c_out_channel __attribute__((depth(10))) __attribute__((io("concat_4c"))); // Channel Tx

channel concat_4b_struct concat_4c_in_b0_channel __attribute__((depth(10))); // internal channel Branch 1
channel concat_4b_struct concat_4c_in_b1_channel __attribute__((depth(10))); // internal channel Branch 2
channel concat_4b_struct concat_4c_in_b2_channel __attribute__((depth(10))); // internal channel Branch 3
channel concat_4b_struct concat_4c_in_b3_channel __attribute__((depth(10))); // internal channel Branch 4

//Auto run kernel to feed 3b results to 4 branches

__kernel void feeder_4c()
{
    for (int i = 0; i < 12544; i++)
    {
        struct concat_4b_buffer input = read_channel_intel(concat_4c_in_channel);
        write_channel_intel(concat_4c_in_b0_channel, input);
        write_channel_intel(concat_4c_in_b1_channel, input);
        write_channel_intel(concat_4c_in_b2_channel, input);
        write_channel_intel(concat_4c_in_b3_channel, input);
    }
}

__kernel void Mixed_4c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4b_buffer in;
        in = read_channel_intel(concat_4c_in_b0_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4b_out_buffer[k];
        }
    }

    for (int ff = 0; ff < 160; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_4c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4b_buffer in;
        in = read_channel_intel(concat_4c_in_b1_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4b_out_buffer[k];
        }
    }

    for (int ff = 0; ff < 112; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? +compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
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
    for (int ff = 0; ff < 224; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 112; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 112) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.0;
                //printf("\t 4c  b1  %d --- %f \n",((((ff * 14) + yy) * 14) + xx),compute[((((ff * 14) + yy) * 14) + xx)]);
            }
        }
    }
}

__kernel void Mixed_4c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4b_buffer in;
        in = read_channel_intel(concat_4c_in_b2_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4b_out_buffer[k];
        }
    }
    for (int ff = 0; ff < 24; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
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
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 24; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 24) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_4c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    //Read Input from IO channel
    float maxInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4b_buffer in;
        in = read_channel_intel(concat_4c_in_b3_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            maxInput[(i * 8) + k] = in.concat_4b_out_buffer[k];
        }
    }
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
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? maxInput[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_4c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Mixed_4c_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    //struct to store 256 bits of data
    struct concat_4c_buffer out;
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        //T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((75264 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -75264)] : (float)((31360 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -31360)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        float result = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((75264 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -75264)] : (float)((31360 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -31360)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        //printf("\t 4c %d - %d --- %f \n",ax0_ax1_fused_ax2_fused_ax3_fused_inner,ax0_ax1_fused_ax2_fused_ax3_fused_inner%8,result);
        out.concat_4c_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_4c_out_channel, out);
        }
    }
}