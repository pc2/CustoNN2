/**
 * 8th Inception module - 5a to mixed_5b_concat
 */
//enable channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_4f_buffer
    {
        float concat_4f_out_buffer[8];
    } concat_4f_struct;

typedef struct concat_5b_buffer
    {
        float concat_5b_out_buffer[8];
    } concat_5b_struct;

// IO Channels for inception 4f to 5a
channel concat_4f_struct concat_5a_in_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_4f_struct concat_5a_in_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_input_ch1"))); // Channel Rx
channel concat_4f_struct concat_5a_in_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_input_ch2"))); // Channel Rx
channel concat_4f_struct concat_5a_in_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_input_ch3"))); // Channel Rx

channel concat_5b_struct concat_5b_out_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel concat_5b_struct concat_5b_out_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_output_ch1"))); // Channel Tx
channel concat_5b_struct concat_5b_out_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_output_ch2"))); // Channel Tx
channel concat_5b_struct concat_5b_out_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_output_ch3"))); // Channel Tx

channel concat_4f_struct concat_5a_in_max_channel __attribute__((depth(32))); // internal channel maxpool

//internal channels
//branch 5a
channel float maxpool_5a_out_channel1 __attribute__((depth(32)));
channel float maxpool_5a_out_channel2 __attribute__((depth(32)));
channel float maxpool_5a_out_channel3 __attribute__((depth(32)));
channel float maxpool_5a_out_channel4 __attribute__((depth(32)));

//branch 0
channel float conv1_5b_out_b0_channel __attribute__((depth(32)));
//branch 1
channel float conv2_1_5b_out_b1_channel __attribute__((depth(32)));
channel float padding_5b_out_b1_channel __attribute__((depth(32)));
channel float conv2_2_5b_out_b1_channel __attribute__((depth(32)));

//branch 2
channel float conv3_1_5b_out_b2_channel __attribute__((depth(32)));
channel float padding_5b_out_b2_channel __attribute__((depth(32)));
channel float conv3_2_5b_out_b2_channel __attribute__((depth(32)));

//branch 3
channel float maxpool_5b_out_b3_channel __attribute__((depth(32)));
channel float conv4_1_5b_out_b3_channel __attribute__((depth(32)));

channel concat_4f_struct concat_5a_in_b0_channel __attribute__((depth(10))); // internal channel maxpool
//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_5a(unsigned int route_from)
{
    for (int i = 0; i < 20384; i++)
    {
        struct concat_4f_buffer input;
        if (route_from == 0)
        {
            input = read_channel_intel(concat_5a_in_channel_0);
        }
        else if (route_from == 1)
        {
            input = read_channel_intel(concat_5a_in_channel_1);
        }
        else if (route_from == 2)
        {
            input = read_channel_intel(concat_5a_in_channel_2);
        }
        else
        {
            input = read_channel_intel(concat_5a_in_channel_3);
        }
        
        write_channel_intel(concat_5a_in_max_channel, input);
    }
}

__kernel void MaxPool_5a_2x2_MaxPool()
{
    //Read Input from IO channel
    float maxInput[163072];
    // 163072/8 = 20384
    
    for (int i = 0; i < 20384; i++)
    {
        //struct to store 256 bits of data
        struct concat_4f_buffer in;
        in = read_channel_intel(concat_5a_in_max_channel);
        
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            maxInput[(i * 8) + k] = in.concat_4f_out_buffer[k];
        }
    }
#pragma loop_coalesce
    for (int ax1 = 0; ax1 < 832; ++ax1)
    {
        for (int ax2 = 0; ax2 < 7; ++ax2)
        {
            for (int ax3 = 0; ax3 < 7; ++ax3)
            {
                float tensor = -3.402823e+38f;
#pragma unroll
                for (int rv = 0; rv < 2; ++rv)
                {
                    float temp_rv1[2];
#pragma unroll
                    for (int rv1 = 0; rv1 < 2; ++rv1)
                    {
                        temp_rv1[rv1] = max(tensor, maxInput[((((((((ax1 * 7) + ax2) * 2) + rv) * 7) + ax3) * 2) + rv1)]);
                    }
#pragma unroll
                    for (int rv1 = 0; rv1 < 2; ++rv1)
                    {
                        tensor = max(tensor, temp_rv1[rv1]);
                    }
                    
                }
                write_channel_intel(maxpool_5a_out_channel1, tensor);
                write_channel_intel(maxpool_5a_out_channel2, tensor);
                write_channel_intel(maxpool_5a_out_channel3, tensor);
                write_channel_intel(maxpool_5a_out_channel4, tensor);
            }
        }
    }
}

__kernel void Mixed_5b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[832 * 7 * 7];
    
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5a_out_channel1);
    }
#pragma loop_coalesce
    for (int ff = 0; ff < 256; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float lc_input2 = input2[ff];
                float lc_input1;
                float temp_0 = lc_input2;
                float temp_rc[832];
#pragma unroll
                for (int rc = 0; rc < 832; ++rc)
                {
                    lc_input1=input1[((ff * 832) + rc)];
                    temp_rc[rc] += (input0[((((rc * 7) + yy) * 7) + xx)] * lc_input1);
                }
#pragma unroll
                for (int rc = 0; rc < 832; ++rc){
                    temp_0 += temp_rc[rc];
                }
                
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_5b_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[832 * 7 * 7];
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5a_out_channel2);
    }
#pragma loop_coalesce
    for (int ff = 0; ff < 160; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                float temp_rc[832];
#pragma unroll
                
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_rc[rc] += (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
#pragma unroll
                for (int rc = 0; rc < 832; ++rc){
                    temp_0 += temp_rc[rc];
                }
                
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_5b_out_b1_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[160 * 7 * 7];
    for (int i = 0; i < 160 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(conv2_1_5b_out_b1_channel);
    }
    
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 12960; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_5b_out_b1_channel, (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f));
    }
}
__kernel void Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[12960];
    for (int i = 0; i < 12960; i++)
    {
        input0[i] = read_channel_intel(padding_5b_out_b1_channel);
    }
#pragma loop_coalesce
    for (int ff = 0; ff < 320; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
#pragma unroll
                for (int rc = 0; rc < 160; ++rc)
                {
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_rx[3];
#pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_rx[rx] += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input1[((((((ff * 160) + rc) * 3) + ry) * 3) + rx)]);
                        }
#pragma unroll
                        for (int rc = 0; rc < 3; ++rc){
                            temp_0 += temp_rx[rc];
                        }
                    }
                }
                
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_5b_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[832 * 7 * 7];
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5a_out_channel3);
    }
#pragma loop_coalesce
    for (int ff = 0; ff < 32; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                float temp_rc[832];
#pragma unroll
                
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_rc[rc] += (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
#pragma unroll
                for (int rc = 0; rc < 832; ++rc){
                    temp_0 += temp_rc[rc];
                }
                
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_5b_out_b2_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D()
{
    float input0[32 * 7 * 7];
    for (int i = 0; i < 32 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(conv3_1_5b_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 2592; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_5b_out_b2_channel, (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f));
    }
}
__kernel void Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[2592];
    for (int i = 0; i < 2592; i++)
    {
        input0[i] = read_channel_intel(padding_5b_out_b2_channel);
    }
#pragma loop_coalesce
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
#pragma unroll
                for (int rc = 0; rc < 32; ++rc)
                {
#pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_rx[3];
#pragma unroll
                        
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_rx[rx] += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]);
                        }
#pragma unroll
                        for (int rc = 0; rc < 3; ++rc){
                            temp_0 += temp_rx[rc];
                        }
                        
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv3_2_5b_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5b_Branch_3_MaxPool_0a_3x3_MaxPool()
{
    float input0[832 * 7 * 7];
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5a_out_channel4);
    }
#pragma loop_coalesce
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
                    
                    float temp_rv1[3];
#pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        temp_rv1[rv1] = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (8 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (8 - rv1))) ? input0[(((((((ax1 * 7) + ax2) + rv) * 7) + ax3) + rv1) + -8)] : -3.402823e+38f));
                    }
#pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, temp_rv1[rv1]);
                    }
                    
                }
                write_channel_intel(maxpool_5b_out_b3_channel, tensor);
            }
        }
    }
}

__kernel void Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[832 * 7 * 7];
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5b_out_b3_channel);
    }
#pragma loop_coalesce
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                float temp_rc[832];
#pragma unroll
                
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_rc[rc] += (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
#pragma unroll
                for (int rc = 0; rc < 832; ++rc){
                    temp_0 += temp_rc[rc];
                }
                
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv4_1_5b_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5b_concat(unsigned int route_to)
{
    //struct to store 256 bits of data
    struct concat_5b_buffer out;
    
    float input0[256 * 7 * 7];
    for (int i = 0; i < 256 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(conv1_5b_out_b0_channel);
    }
    float input1[320 * 7 * 7];
    for (int i = 0; i < 320 * 7 * 7; i++)
    {
        input1[i] = read_channel_intel(conv2_2_5b_out_b1_channel);
    }
    float input2[128 * 7 * 7];
    for (int i = 0; i < 128 * 7 * 7; i++)
    {
        input2[i] = read_channel_intel(conv3_2_5b_out_b2_channel);
    }
    float input3[128 * 7 * 7];
    for (int i = 0; i < 128 * 7 * 7; i++)
    {
        input3[i] = read_channel_intel(conv4_1_5b_out_b3_channel);
    }
    
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((34496 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -34496)] : (float)((28224 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -28224)] : (float)((12544 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -12544)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_5b_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            if (route_to == 0)
            {
                write_channel_intel(concat_5b_out_channel_0, out);
            }
            else if (route_to == 1)
            {
                write_channel_intel(concat_5b_out_channel_1, out);
            }
            else if (route_to == 2)
            {
                write_channel_intel(concat_5b_out_channel_2, out);
            }
            else if (route_to == 3)
            {
                write_channel_intel(concat_5b_out_channel_3, out);
            }
        }
    }
}

