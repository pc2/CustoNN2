/**
 * 3rd Inception module - 4a to mixed_4b_concat
 */
//enable channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_3c_buffer
{
    float concat_3c_out_buffer[8];
} concat_3c_struct;

typedef struct concat_4b_buffer
{
    float concat_4b_out_buffer[8];
} concat_4b_struct;

// IO Channels for inception 3c to 4a
channel concat_3c_struct concat_4a_in_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_3c_struct concat_4a_in_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_input_ch1"))); // Channel Rx
channel concat_3c_struct concat_4a_in_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_input_ch2"))); // Channel Rx
channel concat_3c_struct concat_4a_in_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_input_ch3"))); // Channel Rx

channel concat_4b_struct concat_4b_out_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel concat_4b_struct concat_4b_out_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_output_ch1"))); // Channel Tx
channel concat_4b_struct concat_4b_out_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_output_ch2"))); // Channel Tx
channel concat_4b_struct concat_4b_out_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_output_ch3"))); // Channel Tx

channel concat_3c_struct concat_4a_in_max_channel __attribute__((depth(32))); // internal channel maxpool

//internal channels
//branch 4a
channel float maxpool_4a_out_channel1 __attribute__((depth(32)));
channel float maxpool_4a_out_channel2 __attribute__((depth(32)));
channel float maxpool_4a_out_channel3 __attribute__((depth(32)));
channel float maxpool_4a_out_channel4 __attribute__((depth(32)));

//branch 0
//channel float conv1_4b_out_b0_channel __attribute__((depth(32)));
//branch 1
channel float conv2_1_4b_out_b1_channel __attribute__((depth(32)));
channel float padding_4b_out_b1_channel __attribute__((depth(32)));
//channel float conv2_2_4b_out_b1_channel __attribute__((depth(32)));

//branch 2
channel float conv3_1_4b_out_b2_channel __attribute__((depth(32)));
channel float padding_4b_out_b2_channel __attribute__((depth(32)));
//channel float conv3_2_4b_out_b2_channel __attribute__((depth(32)));

//branch 3
channel float maxpool_4b_out_b3_channel __attribute__((depth(32)));
channel float conv4_1_4b_out_b3_channel __attribute__((depth(32)));

//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_4a(unsigned int route_from)
{
    printf("Kernel started feeder_4a \n");
    for (int i = 0; i < 47040; i++)
    {
        struct concat_3c_buffer input;
        if (route_from == 0)
        {
            input = read_channel_intel(concat_4a_in_channel_0);
        }
        else if (route_from == 1)
        {
            input = read_channel_intel(concat_4a_in_channel_1);
        }
        else if (route_from == 2)
        {
            input = read_channel_intel(concat_4a_in_channel_2);
        }
        else
        {
            input = read_channel_intel(concat_4a_in_channel_3);
        }

        write_channel_intel(concat_4a_in_max_channel, input);
    }
    printf("Kernel fnshd feeder_4a \n");
}

__kernel void MaxPool_4a_3x3_MaxPool()
{
    printf("Kernel started MaxPool_4a_3x3_MaxPool \n");
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
        }
    }

    for (int ax1 = 0; ax1 < 480; ++ax1)
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
                        tensor = max(tensor, (float)((((ax2 * 2) < (28 - rv)) && ((ax3 * 2) < (28 - rv1))) ? maxInput[((((((((ax1 * 14) + ax2) * 2) + rv) * 14) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_4a_out_channel1, tensor);
                write_channel_intel(maxpool_4a_out_channel2, tensor);
                write_channel_intel(maxpool_4a_out_channel3, tensor);
                write_channel_intel(maxpool_4a_out_channel4, tensor);
            }
        }
    }
    printf("Kernel fnshd MaxPool_4a_3x3_MaxPool \n");
}

__kernel void Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,__global float *restrict input2, __global float *restrict output)
{
    printf("Kernel started Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D \n");

    float input0[480 * 14 * 14];

    for (int i = 0; i < 480 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(maxpool_4a_out_channel1);
    }

    //local memory for biases
    __local float input_bias[192];
#pragma unroll 64
    for (int j = 0; j < 192; j++)
    {
        input_bias[j] = input2[j];
    }

    for (int ff = 0; ff < 192; ++ff)
    {
        //local memory for weights
        float input_weight[480];
#pragma unroll 64
        for (int k = 0; k < 480; k++)
        {
            input_weight[k] = input1[((ff * 480) + k)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                // #pragma unroll 4
                for (int rc = 0; rc < 480; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weight[(rc)]);
                }

                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
               // write_channel_intel(conv1_4b_out_b0_channel, temp_0);
                output[((((ff * 14) + yy) * 14) + xx)] = temp_0 ;
            }
        }
    }
     printf("Kernel fnshd Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Mixed_4b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    printf("Kernel started Mixed_4b_Branch_1_Conv2d_0a_1x1_Conv2D \n");
    float input0[480 * 14 * 14];
    for (int i = 0; i < 480 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(maxpool_4a_out_channel2);
    }
    //local memory for biases
    __local float input_bias[96];
#pragma unroll 32
    for (int j = 0; j < 96; j++)
    {
        input_bias[j] = input2[j];
    }

    for (int ff = 0; ff < 96; ++ff)
    {
        //local memory for weights
        float input_weight[480];
#pragma unroll 64
        for (int k = 0; k < 480; k++)
        {
            input_weight[k] = input1[((ff * 480) + k)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                //  #pragma unroll 4
                for (int rc = 0; rc < 480; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_4b_out_b1_channel, temp_0);
            }
        }
    }
     printf("Kernel fnshd Mixed_4b_Branch_1_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Padding_Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D()
{
     printf("Kernel started Padding_Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D \n");
    float input0[96 * 14 * 14];
    for (int i = 0; i < 96 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv2_1_4b_out_b1_channel);
    }

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 24576; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4b_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
     printf("Kernel fnshd Padding_Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,__global float *restrict input2, __global float *restrict output)
{
     printf("Kernel started Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D \n");
    float input0[24576];
    for (int i = 0; i < 24576; i++)
    {
        input0[i] = read_channel_intel(padding_4b_out_b1_channel);
    }
    //local memory for biases
    __local float input_bias[208];
#pragma unroll 64
    for (int j = 0; j < 208; j++)
    {
        input_bias[j] = input2[j];
    }

    for (int ff = 0; ff < 208; ++ff)
    {
        //local memory for weights
        float input_weight[3 * 3 * 96];
#pragma unroll 32
        for (int k = 0; k < 3 * 3 * 96; k++)
        {
            input_weight[k] = input1[((ff * 3 * 3 * 96) + k)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                //  #pragma unroll
                for (int rc = 0; rc < 96; ++rc)
                {
                    float temp_2 = 0.0;
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
#pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weight[(((((rc)*3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                //write_channel_intel(conv2_2_4b_out_b1_channel, temp_0);
                 output[((((ff * 14) + yy) * 14) + xx)] = temp_0 ;
            }
        }
    }
     printf("Kernel fnshd Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_4b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
     printf("Kernel started Mixed_4b_Branch_2_Conv2d_0a_1x1_Conv2D \n");
    float input0[480 * 14 * 14];
    for (int i = 0; i < 480 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(maxpool_4a_out_channel3);
    }
    //local memory for biases
    __local float input_bias[16];
#pragma unroll
    for (int j = 0; j < 16; j++)
    {
        input_bias[j] = input2[j];
    }

    for (int ff = 0; ff < 16; ++ff)
    {
        //local memory for weights
        float input_weight[480];
#pragma unroll 8
        for (int k = 0; k < 480; k++)
        {
            input_weight[k] = input1[((ff * 480) + k)];
        }

        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                //  #pragma unroll 4
                for (int rc = 0; rc < 480; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_4b_out_b2_channel, temp_0);
            }
        }
    }
     printf("Kernel fnshd Mixed_4b_Branch_2_Conv2d_0a_1x1_Conv2D \n");
}
__kernel void Padding_Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D()
{
     printf("Kernel started Padding_Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D \n");
    float input0[16 * 14 * 14];
    for (int i = 0; i < 16 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv3_1_4b_out_b2_channel);
    }

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 4096; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4b_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }

     printf("Kernel fnshd Padding_Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D \n");
}
__kernel void Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,__global float *restrict input2 , __global float *restrict output)
{
    printf("Kernel started Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D \n");
    float input0[4096];
    for (int i = 0; i < 4096; i++)
    {
        input0[i] = read_channel_intel(padding_4b_out_b2_channel);
    }
    //local memory for biases
    __local float input_bias[48];
#pragma unroll 4
    for (int j = 0; j < 48; j++)
    {
        input_bias[j] = input2[j];
    }

    for (int ff = 0; ff < 48; ++ff)
    {
        //local memory for weights
        float input_weight[3 * 3 * 16];
#pragma unroll 4
        for (int k = 0; k < 3 * 3 * 16; k++)
        {
            input_weight[k] = input1[((ff * 3 * 3 * 16) + k)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                //   #pragma unroll
                for (int rc = 0; rc < 16; ++rc)
                {
                    float temp_2 = 0.0;
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
#pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weight[(((((rc)*3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
               // write_channel_intel(conv3_2_4b_out_b2_channel, temp_0);
                output[((((ff * 14) + yy) * 14) + xx)] = temp_0 ;

            }
        }
    }
     printf("Kernel fnshd Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_4b_Branch_3_MaxPool_0a_3x3_MaxPool()
{
      printf("Kernel started Mixed_4b_Branch_3_MaxPool_0a_3x3_MaxPool \n");
    for (int ax1 = 0; ax1 < 480; ++ax1)
    {
        float input0[14 * 14];
        for (int i = 0; i < 14 * 14; i++)
        {
            input0[i] = read_channel_intel(maxpool_4a_out_channel4);
        }

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
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input0[((((((ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_4b_out_b3_channel, tensor);
            }
        }
    }
      printf("Kernel fnshd Mixed_4b_Branch_3_MaxPool_0a_3x3_MaxPool \n");
}

__kernel void Mixed_4b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2, __global float *restrict output )
{
     printf("Kernel started Mixed_4b_Branch_3_Conv2d_0b_1x1_Conv2D \n");
    float input0[480 * 14 * 14];
    for (int i = 0; i < 480 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(maxpool_4b_out_b3_channel);
    }
    //local memory for biases
    __local float input_bias[64];
#pragma unroll 4
    for (int j = 0; j < 64; j++)
    {
        input_bias[j] = input2[j];
    }

    for (int ff = 0; ff < 64; ++ff)
    {
        //local memory for weights
        float input_weight[480];
#pragma unroll 8
        for (int k = 0; k < 480; k++)
        {
            input_weight[k] = input1[((ff * 480) + k)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                //  #pragma unroll 8
                for (int rc = 0; rc < 480; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
               // write_channel_intel(conv4_1_4b_out_b3_channel, temp_0);
               output[((((ff * 14) + yy) * 14) + xx)] = temp_0 ;
            }
        }
    }
     printf("Kernel fnshd Mixed_4b_Branch_3_Conv2d_0b_1x1_Conv2D \n");
}

__kernel void Mixed_4b_concat(unsigned int route_to, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    printf("Kernel started Mixed_4b_concat \n");
    //struct to store 256 bits of data
    struct concat_4b_buffer out;
    
    /*
    float input0[192 * 14 * 14];
    for (int i = 0; i < 192 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv1_4b_out_b0_channel);
    }
    float input1[208 * 14 * 14];
    for (int i = 0; i < 208 * 14 * 14; i++)
    {
        input1[i] = read_channel_intel(conv2_2_4b_out_b1_channel);
    }
    float input2[48 * 14 * 14];
    for (int i = 0; i < 48 * 14 * 14; i++)
    {
        input2[i] = read_channel_intel(conv3_2_4b_out_b2_channel);
    }
    float input3[64 * 14 * 14];
    for (int i = 0; i < 64 * 14 * 14; i++)
    {
        input3[i] = read_channel_intel(conv4_1_4b_out_b3_channel);
    }
    */

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_4b_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            if (route_to == 0)
            {
                write_channel_intel(concat_4b_out_channel_0, out);
            }
            else if (route_to == 1)
            {
                write_channel_intel(concat_4b_out_channel_1, out);
            }
            else if (route_to == 2)
            {
                write_channel_intel(concat_4b_out_channel_2, out);
            }
            else if (route_to == 3)
            {
                write_channel_intel(concat_4b_out_channel_3, out);
            }
        }
    }
     printf("Kernel fnshd  Mixed_4b_concat \n");
}
