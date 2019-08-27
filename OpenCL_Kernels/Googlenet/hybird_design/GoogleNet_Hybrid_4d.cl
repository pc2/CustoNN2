//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_4c_buffer
{
    float concat_4c_out_buffer[8];
} concat_4c_struct;

typedef struct concat_4d_buffer
{
    float concat_4d_out_buffer[8];
} concat_4d_struct;

// IO Channels for inception 4c to 4d
channel concat_4c_struct concat_4d_in_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_4c_struct concat_4d_in_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_input_ch1"))); // Channel Rx
channel concat_4c_struct concat_4d_in_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_input_ch2"))); // Channel Rx
channel concat_4c_struct concat_4d_in_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_input_ch3"))); // Channel Rx

channel concat_4d_struct concat_4d_out_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel concat_4d_struct concat_4d_out_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_output_ch1"))); // Channel Tx
channel concat_4d_struct concat_4d_out_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_output_ch2"))); // Channel Tx
channel concat_4d_struct concat_4d_out_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_output_ch3"))); // Channel Tx

channel concat_4c_struct concat_4d_in_b0_channel __attribute__((depth(32))); // internal channel Branch 1
channel concat_4c_struct concat_4d_in_b1_channel __attribute__((depth(32))); // internal channel Branch 2
channel concat_4c_struct concat_4d_in_b2_channel __attribute__((depth(32))); // internal channel Branch 3
channel concat_4c_struct concat_4d_in_b3_channel __attribute__((depth(32))); // internal channel Branch 4

//internal channels
//branch 0
//channel float conv1_4d_out_b0_channel __attribute__((depth(32)));
//branch 1
channel float conv2_1_4d_out_b1_channel __attribute__((depth(32)));
channel float padding_4d_out_b1_channel __attribute__((depth(32)));
//channel float conv2_2_4d_out_b1_channel __attribute__((depth(32)));
//branch 2
channel float conv3_1_4d_out_b2_channel __attribute__((depth(32)));
channel float padding_4d_out_b2_channel __attribute__((depth(32)));
//channel float conv3_2_4d_out_b2_channel __attribute__((depth(32)));
//branch 3
channel float maxpool_4d_out_b3_channel __attribute__((depth(32)));
//channel float conv4_1_4d_out_b3_channel __attribute__((depth(32)));

//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_4d(unsigned int route_from)
{
    printf("Kernel started feeder_4d \n");
    for (int i = 0; i < 12544; i++)
    {
        struct concat_4c_buffer input;
        if (route_from == 0)
        {
            input = read_channel_intel(concat_4d_in_channel_0);
        }
        else if (route_from == 1)
        {
            input = read_channel_intel(concat_4d_in_channel_1);
        }
        else if (route_from == 2)
        {
            input = read_channel_intel(concat_4d_in_channel_2);
        }
        else // if (route_from == 3)
        {
            input = read_channel_intel(concat_4d_in_channel_3);
        }

        write_channel_intel(concat_4d_in_b0_channel, input);
        write_channel_intel(concat_4d_in_b1_channel, input);
        write_channel_intel(concat_4d_in_b2_channel, input);
        write_channel_intel(concat_4d_in_b3_channel, input);
    }
    printf("Kernel fnshd feeder_4d \n");
}

__kernel void Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, constant float *restrict input2, __global float *restrict output)
{
    printf("Kernel started Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D \n");
    //Read Input from IO channel
    float input0[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4c_buffer in;
        in = read_channel_intel(concat_4d_in_b0_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_4c_out_buffer[k];
        }
    }

    //Local memory for Biases:
    __local float input_bias[128];
#pragma unroll 64
    for (int b = 0; b < 128; b++)
    {
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 128; ++ff)
    {
        //Local weights
        float input_weights[512];
#pragma unroll 32
        for (int m = 0; m < 512; m++)
        {
            input_weights[m] = input1[((ff * 512) + m)];
        }

        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0;
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                //write_channel_intel(conv1_4d_out_b0_channel, temp_0);
                output[((((ff * 14) + yy) * 14) + xx)] = temp_0;
            }
        }
    }
    printf("Kernel fnshd Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Mixed_4d_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, constant float *restrict input2)
{
    printf("Kernel started Mixed_4d_Branch_1_Conv2d_0a_1x1_Conv2D \n");
    //Read Input from IO channel
    float input0[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4c_buffer in;
        in = read_channel_intel(concat_4d_in_b1_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_4c_out_buffer[k];
        }
    }
    //Local memory for Biases:
    __local float input_bias[128];
#pragma unroll 64
    for (int b = 0; b < 128; b++)
    {
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 128; ++ff)
    {
        //Local weights
        float input_weights[512];
#pragma unroll 32
        for (int m = 0; m < 512; m++)
        {
            input_weights[m] = input1[((ff * 512) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_4d_out_b1_channel, temp_0);
            }
        }
    }
    printf("Kernel fnshd Mixed_4d_Branch_1_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Padding_Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    printf("Kernel started Padding_Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D \n");
    float input0[128 * 14 * 14];
    for (int i = 0; i < 128 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv2_1_4d_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 32768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4d_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
    printf("Kernel fnshd Padding_Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D \n");
}
__kernel void Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2, 
                                                     __global float *restrict output)
{
    printf("Kernel started Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D \n");
    float input0[32768];
    for (int i = 0; i < 32768; i++)
    {
        input0[i] = read_channel_intel(padding_4d_out_b1_channel);
    }

    //Local memory for Biases:
    __local float input_bias[256];
#pragma unroll 32
    for (int b = 0; b < 256; b++)
    {
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 256; ++ff)
    {
        //Local weights
        float input_weights[3 * 3 * 128];
#pragma unroll 32
        for (int m = 0; m < 3 * 3 * 128; m++)
        {
            input_weights[m] = input1[((ff * 3 * 3 * 128) + m)];
        }

        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 128; ++rc)
                {
                    float temp_2 = 0.0;
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
#pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[(((((rc)*3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }

                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                //write_channel_intel(conv2_2_4d_out_b1_channel, temp_0);
                output[((((ff * 14) + yy) * 14) + xx)] = temp_0;
            }
        }
    }
    printf("Kernel fnshd Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_4d_Branch_2_Conv2d_0a_1x1_Conv2D(__global const float *restrict input1,
                                                     constant float *restrict input2)
{
    printf("Kernel started Mixed_4d_Branch_2_Conv2d_0a_1x1_Conv2D \n");
    //Read Input from IO channel
    float input0[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4c_buffer in;
        in = read_channel_intel(concat_4d_in_b2_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_4c_out_buffer[k];
        }
    }

    //Local memory for Biases:
    __local float input_bias[24];
#pragma unroll 32
    for (int b = 0; b < 24; b++)
    {
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 24; ++ff)
    {
        //Local weights
        float input_weights[512];
#pragma unroll 32
        for (int m = 0; m < 512; m++)
        {
            input_weights[m] = input1[((ff * 512) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_4d_out_b2_channel, temp_0);
            }
        }
    }
     printf("Kernel fnshd Mixed_4d_Branch_2_Conv2d_0a_1x1_Conv2D \n");
}
__kernel void Padding_Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D()
{
     printf("Kernel started Padding_Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D \n");
    float input0[24 * 14 * 14];
    for (int i = 0; i < 24 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv3_1_4d_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4d_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
     printf("Kernel fnshd Padding_Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D \n");
}
__kernel void Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2 , __global float *restrict output)
{
     printf("Kernel started Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D \n");
    float input0[6144];
    for (int i = 0; i < 6144; i++)
    {
        input0[i] = read_channel_intel(padding_4d_out_b2_channel);
    }

    //Local memory for Biases:
    __local float input_bias[64];
#pragma unroll 32
    for (int b = 0; b < 64; b++)
    {
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 64; ++ff)
    {
        //Local weights
        float input_weights[3 * 3 * 24];
#pragma unroll 32
        for (int m = 0; m < 3 * 3 * 24; m++)
        {
            input_weights[m] = input1[((ff * 3 * 3 * 24) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 24; ++rc)
                {
                    float temp_2 = 0.0;
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
#pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[(((((rc)*3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                //write_channel_intel(conv3_2_4d_out_b2_channel, temp_0);
                 output[((((ff * 14) + yy) * 14) + xx)] = temp_0;
            }
        }
    }
     printf("Kernel fnshd Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_4d_Branch_3_MaxPool_0a_3x3_MaxPool()
{
     printf("Kernel started Mixed_4d_Branch_3_MaxPool_0a_3x3_MaxPool \n");
    //Read Input from IO channel
    float maxInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4c_buffer in;
        in = read_channel_intel(concat_4d_in_b3_channel);

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            maxInput[(i * 8) + k] = in.concat_4c_out_buffer[k];
        }
    }

    for (int ax1 = 0; ax1 < 512; ++ax1)
    {
        for (int ax2 = 0; ax2 < 14; ++ax2)
        {
            for (int ax3 = 0; ax3 < 14; ++ax3)
            {
                float tensor = -3.402823e+38f;
#pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
#pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? maxInput[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_4d_out_b3_channel, tensor);
            }
        }
    }
      printf("Kernel fnshd Mixed_4d_Branch_3_MaxPool_0a_3x3_MaxPool \n");
}

__kernel void Mixed_4d_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1,constant float *restrict input2, __global float *restrict output)
{

     printf("Kernel started Mixed_4d_Branch_3_Conv2d_0b_1x1_Conv2D \n");
    float input0[512 * 14 * 14];
    for (int i = 0; i < 512 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(maxpool_4d_out_b3_channel);
    }
    //Local memory for Biases:
    __local float input_bias[64];
#pragma unroll 64
    for (int b = 0; b < 64; b++)
    {
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 64; ++ff)
    {
        //Local weights
        float input_weights[512];
#pragma unroll 32
        for (int m = 0; m < 512; m++)
        {
            input_weights[m] = input1[((ff * 512) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                //#pragma unroll 8
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
               // write_channel_intel(conv4_1_4d_out_b3_channel, temp_0);
               output[((((ff * 14) + yy) * 14) + xx)] = temp_0;
            }
        }
    }
     printf("Kernel fnshd Mixed_4d_Branch_3_Conv2d_0b_1x1_Conv2D \n");
}

__kernel void Mixed_4d_concat(unsigned int route_to, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    printf("Kernel started Mixed_4d_concat \n");
    //struct to store 256 bits of data
    struct concat_4d_buffer out;
    /*
    float input0[128 * 14 * 14];
    for (int i = 0; i < 128 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv1_4d_out_b0_channel);
    }
    float input1[256 * 14 * 14];
    for (int i = 0; i < 256 * 14 * 14; i++)
    {
        input1[i] = read_channel_intel(conv2_2_4d_out_b1_channel);
    }
    float input2[64 * 14 * 14], input3[64 * 14 * 14];
    for (int i = 0; i < 64 * 14 * 14; i++)
    {
        input2[i] = read_channel_intel(conv3_2_4d_out_b2_channel);
        input3[i] = read_channel_intel(conv4_1_4d_out_b3_channel);
    }
    */

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((75264 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -75264)] : (float)((25088 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -25088)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_4d_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            if (route_to == 0)
            {
                write_channel_intel(concat_4d_out_channel_0, out);
            }
            else if (route_to == 1)
            {
                write_channel_intel(concat_4d_out_channel_1, out);
            }
            else if (route_to == 2)
            {
                write_channel_intel(concat_4d_out_channel_2, out);
            }
            else if (route_to == 3)
            {
                write_channel_intel(concat_4d_out_channel_3, out);
            }
        }
    }
    printf("Kernel fnshd Mixed_4d_concat \n");
}
