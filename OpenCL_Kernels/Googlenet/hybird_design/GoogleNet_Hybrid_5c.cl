/**
 * 9th Inception module - Inception 5c
 */
//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_5b_buffer
{
    float concat_5b_out_buffer[8];
} concat_5b_struct;

// IO Channels for inception 5b to 5c
channel concat_5b_struct concat_5c_in_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_5b_struct concat_5c_in_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_input_ch1"))); // Channel Rx
channel concat_5b_struct concat_5c_in_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_input_ch2"))); // Channel Rx
channel concat_5b_struct concat_5c_in_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_input_ch3"))); // Channel Rx

channel concat_5b_struct concat_5c_in_b0_channel __attribute__((depth(32))); // internal channel Branch 1
channel concat_5b_struct concat_5c_in_b1_channel __attribute__((depth(32))); // internal channel Branch 2
channel concat_5b_struct concat_5c_in_b2_channel __attribute__((depth(32))); // internal channel Branch 3
channel concat_5b_struct concat_5c_in_b3_channel __attribute__((depth(32))); // internal channel Branch 4

//internal channles
//branch 0
//channel float conv1_5c_out_b0_channel __attribute__((depth(32)));

//branch 1
channel float conv2_1_5c_out_b1_channel __attribute__((depth(32)));
channel float padding_5c_out_b1_channel __attribute__((depth(32)));
//channel float conv2_2_5c_out_b1_channel __attribute__((depth(32)));

//branch 2
channel float conv3_1_5c_out_b2_channel __attribute__((depth(32)));
channel float padding_5c_out_b2_channel __attribute__((depth(32)));
//channel float conv3_2_5c_out_b2_channel __attribute__((depth(32)));

//branch 3
channel float maxpool_5c_out_b3_channel __attribute__((depth(32)));
//channel float conv4_1_5c_out_b3_channel __attribute__((depth(32)));

//concat
channel float concat_5c_out_channel __attribute__((depth(32)));

//avgpool
channel float avgpool_out_channel __attribute__((depth(32)));

//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_5c(unsigned int route_from)
{
    printf("Kernel started feeder_5c \n");
    for (int i = 0; i < 5096; i++)
    {
        struct concat_5b_buffer input;
        if (route_from == 0)
        {
            input = read_channel_intel(concat_5c_in_channel_0);
        }
        else if (route_from == 1)
        {
            input = read_channel_intel(concat_5c_in_channel_1);
        }
        else if (route_from == 2)
        {
            input = read_channel_intel(concat_5c_in_channel_2);
        }
        else
        {
            input = read_channel_intel(concat_5c_in_channel_3);
        }

        write_channel_intel(concat_5c_in_b0_channel, input);
        write_channel_intel(concat_5c_in_b1_channel, input);
        write_channel_intel(concat_5c_in_b2_channel, input);
        write_channel_intel(concat_5c_in_b3_channel, input);
    }
    printf("Kernel fnshd feeder_5c \n");
}

__kernel void Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2, __global float *restrict output)
{
    printf("Kernel started Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D \n");
    //Read Input from IO channel
    float convInput[40768];
    // 40768/8 = 5096
    for (int i = 0; i < 5096; i++)
    {
        //struct to store 256 bits of data
        struct concat_5b_buffer in;
        in = read_channel_intel(concat_5c_in_b0_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_5b_out_buffer[k];
        }
    }
    __local float input_bias[384];
#pragma unroll 128
    for (int b = 0; b < 384; b++)
    {
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 384; ++ff)
    {
        float input_weights[832];
#pragma unroll 128
        for (int w = 0; w < 832; w++)
        {
            input_weights[w] = input1[((ff * 832) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_1 += (convInput[((((rc * 7) + yy) * 7) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                //write_channel_intel(conv1_5c_out_b0_channel, temp_0);
                output[((((ff * 7) + yy) * 7) + xx)] = temp_0;
            }
        }
    }
    printf("Kernel fnshd Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("Kernel started Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D \n");
    //Read Input from IO channel
    float convInput[40768];
    // 40768/8 = 5096

    for (int i = 0; i < 5096; i++)
    {
        //struct to store 256 bits of data
        struct concat_5b_buffer in;
        in = read_channel_intel(concat_5c_in_b1_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_5b_out_buffer[k];
        }
    }
    __local float input_bias[192];
#pragma unroll 128
    for (int b = 0; b < 192; b++)
    {
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 192; ++ff)
    {
        float input_weights[832];
#pragma unroll 128
        for (int w = 0; w < 832; w++)
        {
            input_weights[w] = input1[((ff * 832) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_1 += (convInput[((((rc * 7) + yy) * 7) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_5c_out_b1_channel, temp_0);
            }
        }
    }
    printf("Kernel fnshd Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Padding_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    printf("Kernel started Padding_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D \n");
    float input0[192 * 7 * 7];
    for (int i = 0; i < 192 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(conv2_1_5c_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 15552; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_5c_out_b1_channel, (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f));
    }
    printf("Kernel fnshd Padding_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D \n");
}
__kernel void Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2,__global float *restrict output)
{
    printf("Kernel started Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D \n");
    float input0[15552];
    for (int i = 0; i < 15552; i++)
    {
        input0[i] = read_channel_intel(padding_5c_out_b1_channel);
    }
    __local float input_bias[384];
#pragma unroll 128
    for (int b = 0; b < 384; b++)
    {
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 384; ++ff)
    {
        float input_weights[192 * 3 * 3];
#pragma unroll 128
        for (int w = 0; w < 192 * 3 * 3; w++)
        {
            input_weights[w] = input1[((ff * 192 * 3 * 3) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 192; ++rc)
                {
                    float temp_2 = 0.0;
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
#pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input_weights[(((((rc)*3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                //write_channel_intel(conv2_2_5c_out_b1_channel, temp_0);
                output[((((ff * 7) + yy) * 7) + xx)] = temp_0;
            }
        }
    }
    printf("Kernel fnshd Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("Kernel started Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D \n");
    //Read Input from IO channel
    float convInput[40768];
    // 40768/8 = 5096

    for (int i = 0; i < 5096; i++)
    {
        //struct to store 256 bits of data
        struct concat_5b_buffer in;
        in = read_channel_intel(concat_5c_in_b2_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_5b_out_buffer[k];
        }
    }
    __local float input_bias[48];
#pragma unroll 32
    for (int b = 0; b < 48; b++)
    {
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 48; ++ff)
    {
        float input_weights[832];
#pragma unroll 128
        for (int w = 0; w < 832; w++)
        {
            input_weights[w] = input1[((ff * 832) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_1 += (convInput[((((rc * 7) + yy) * 7) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_5c_out_b2_channel, temp_0);
            }
        }
    }
    printf("Kernel fnshd Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Padding_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    printf("Kernel started Padding_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D \n");
    float input0[48 * 7 * 7];
    for (int i = 0; i < 48 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(conv3_1_5c_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 3888; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_5c_out_b2_channel, (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f));
    }
    printf("Kernel fnshd Padding_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D \n");
}
__kernel void Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2, __global float *restrict output)
{
    printf("Kernel started Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D \n");
    float input0[3888];
    for (int i = 0; i < 3888; i++)
    {
        input0[i] = read_channel_intel(padding_5c_out_b2_channel);
    }
    __local float input_bias[128];
#pragma unroll 32
    for (int b = 0; b < 128; b++)
    {
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 128; ++ff)
    {
        float input_weights[48 * 3 * 3];
#pragma unroll 32
        for (int w = 0; w < 48 * 3 * 3; w++)
        {
            input_weights[w] = input1[((ff * 48 * 3 * 3) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 48; ++rc)
                {
                    float temp_2 = 0.0;
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
#pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input_weights[(((((rc)*3) + ry) * 3) + rx)]);
                        }

                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                //write_channel_intel(conv3_2_5c_out_b2_channel, temp_0);
                output[((((ff * 7) + yy) * 7) + xx)] = temp_0;
            }
        }
    }
    printf("Kernel fnshd Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_5c_Branch_3_MaxPool_0a_3x3_MaxPool()
{
    printf("Kernel started Mixed_5c_Branch_3_MaxPool_0a_3x3_MaxPool \n");
    //Read Input from IO channel
    float maxInput[40768];
    // 40768/8 = 5096

    for (int i = 0; i < 5096; i++)
    {
        //struct to store 256 bits of data
        struct concat_5b_buffer in;
        in = read_channel_intel(concat_5c_in_b3_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            maxInput[(i * 8) + k] = in.concat_5b_out_buffer[k];
        }
    }

    for (int ax1 = 0; ax1 < 832; ++ax1)
    {
        for (int ax2 = 0; ax2 < 7; ++ax2)
        {
            for (int ax3 = 0; ax3 < 7; ++ax3)
            {
                float tensor = -3.402823e+38f;
#pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
#pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (8 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (8 - rv1))) ? maxInput[(((((((ax1 * 7) + ax2) + rv) * 7) + ax3) + rv1) + -8)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_5c_out_b3_channel, tensor);
            }
        }
    }
    printf("Kernel fnshd Mixed_5c_Branch_3_MaxPool_0a_3x3_MaxPool \n");
}

__kernel void Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2,__global float *restrict output)
{
    printf("Kernel started Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D \n");
    float input0[832 * 7 * 7];
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5c_out_b3_channel);
    }
    __local float input_bias[128];
#pragma unroll 32
    for (int b = 0; b < 128; b++)
    {
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 128; ++ff)
    {
        float input_weights[832];
#pragma unroll 32
        for (int w = 0; w < 832; w++)
        {
            input_weights[w] = input1[((ff * 832) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_1 += (input0[((((rc * 7) + yy) * 7) + xx)] * input_weights[(rc)]);
                }

                temp_0 += temp_1;

                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                //write_channel_intel(conv4_1_5c_out_b3_channel, temp_0);
                output[((((ff * 7) + yy) * 7) + xx)] = temp_0;
            }
        }
    }
    printf("Kernel fnshd Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D \n");
}

__kernel void Mixed_5c_concat(__global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    printf("Kernel started Mixed_5c_concat \n");
    //struct to store 256 bits of data
    //struct concat_5c_buffer out;

    /*
    float input0[384 * 7 * 7];
    for (int i = 0; i < 384 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(conv1_5c_out_b0_channel);
    }
    float input1[384 * 7 * 7];
    for (int i = 0; i < 384 * 7 * 7; i++)
    {
        input1[i] = read_channel_intel(conv2_2_5c_out_b1_channel);
    }
    float input2[128 * 7 * 7], input3[128 * 7 * 7];
    for (int i = 0; i < 128 * 7 * 7; i++)
    {
        input2[i] = read_channel_intel(conv3_2_5c_out_b2_channel);
        input3[i] = read_channel_intel(conv4_1_5c_out_b3_channel);
    }
    */

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((43904 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -43904)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : (float)((18816 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -18816)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        write_channel_intel(concat_5c_out_channel, result);
    }
    printf("Kernel fnshd Mixed_5c_concat \n");
}

__kernel void AvgPool_0a_7x7_AvgPool()
{
    printf("Kernel started AvgPool_0a_7x7_AvgPool \n");
    float input0[50176];
    for (int i = 0; i < 50176; i++)
    {
        input0[i] = read_channel_intel(concat_5c_out_channel);
    }
    for (int ax1 = 0; ax1 < 1024; ++ax1)
    {
        float tensor = 0.000000e+00f;
        for (int rv = 0; rv < 7; ++rv)
        {
            for (int rv1 = 0; rv1 < 7; ++rv1)
            {
                tensor = (tensor + (input0[((((ax1 * 7) + rv) * 7) + rv1)] * 2.040816e-02f));
            }
        }
        write_channel_intel(avgpool_out_channel, tensor);
    }
    printf("Kernel fnshd AvgPool_0a_7x7_AvgPool \n");
}

__kernel void Conv2d_0c_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2, __global float *restrict compute)
{
    printf("Kernel started Conv2d_0c_1x1_Conv2D \n");
    float input0[1024];
    for (int i = 0; i < 1024; i++)
    {
        input0[i] = read_channel_intel(avgpool_out_channel);
    }
    __local float input_bias[1001];
#pragma unroll 32
    for (int b = 0; b < 1001; b++)
    {
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 1001; ++ff)
    {
        float input_weights[1024];
#pragma unroll 32
        for (int w = 0; w < 1024; w++)
        {
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
    printf("Kernel fnshd Conv2d_0c_1x1_Conv2D \n");
}
