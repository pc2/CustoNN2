/**
 * 1st Inception module - Conv 1a to Mixed_3b_concat
 */

//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable
//256 bits io channel struct

typedef struct concat_3a_buffer
    {
        float concat_3a_out_buffer[8];
    } concat_3a_struct;
// IO Channels for inception 3b to 3c
channel concat_3a_struct concat_3a_out_channel __attribute__((depth(10))) __attribute__((io("concat_3a"))); // Channel Tx

//branch 1a
channel float padding_1a_out_channel __attribute__((depth(32)));
channel float conv1_1a_out_channel __attribute__((depth(32)));
//branch 2a
channel float maxpool_2a_out_channel __attribute__((depth(32)));
//branch 2b
channel float conv1_2b_out_channel __attribute__((depth(32)));
//branch 2c
channel float padding_2c_out_channel __attribute__((depth(32)));
channel float conv1_2c_out_channel __attribute__((depth(32)));








__kernel void Padding_Conv2d_1a_7x7_Conv2D(__global float *restrict input0)
{
    printf("In Padding_Conv2d_1a_7x7_Conv2D \n");
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 157323; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_1a_out_channel, (float)(((((458 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) < 51754)) && (2 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229) < 226)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) / 229) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52441)) + -1350)] : 0.000000e+00f));
    }
    printf("Done Padding_Conv2d_1a_7x7_Conv2D \n");
}

__kernel void Conv2d_1a_7x7_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("In Conv2d_1a_7x7_Conv2D \n");
    float input0[157323];
    for (int i = 0; i < 157323; i++){
        input0[i] = read_channel_intel(padding_1a_out_channel);
    }
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 112; ++yy)
        {
            for (int xx = 0; xx < 112; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 3; ++rc)
                {
                    for (int ry = 0; ry < 7; ++ry)
                    {
                        for (int rx = 0; rx < 7; ++rx)
                        {
                            temp_0 += (input0[(((((rc * 52441) + (yy * 458)) + (ry * 229)) + (xx * 2)) + rx)] * input1[((((((ff * 3) + rc) * 7) + ry) * 7) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_1a_out_channel, temp_0);
            }
        }
    }
    printf("Done Conv2d_1a_7x7_Conv2D \n");
}

__kernel void MaxPool_2a_3x3_MaxPool()
{
    printf("In MaxPool_2a_3x3_MaxPool \n");
    float input0[64*112*112];
    for (int i = 0; i < 64*112*112; i++){
        input0[i] = read_channel_intel(conv1_1a_out_channel);
    }
    
    for (int ax1 = 0; ax1 < 64; ++ax1)
    {
        for (int ax2 = 0; ax2 < 56; ++ax2)
        {
            for (int ax3 = 0; ax3 < 56; ++ax3)
            {
                float tensor = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((ax2 * 2) < (112 - rv)) && ((ax3 * 2) < (112 - rv1))) ? input0[((((((((ax1 * 56) + ax2) * 2) + rv) * 56) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_2a_out_channel, tensor);
            }
        }
    }
    printf("Done MaxPool_2a_3x3_MaxPool \n");
}

__kernel void Conv2d_2b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("In Conv2d_2b_1x1_Conv2D \n");
    float input0[64*56*56];
    for (int i = 0; i < 64*56*56; i++){
        input0[i] = read_channel_intel(maxpool_2a_out_channel);
    }
     printf("Conv2d_2b_1x1_Conv2D : done reading data from IO \n");
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 56; ++yy)
        {
            for (int xx = 0; xx < 56; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 64; ++rc)
                {
                    temp_0 += (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 64) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_2b_out_channel, temp_0);
            }
        }
    }
    printf("Done Conv2d_2b_1x1_Conv2D \n");
}

__kernel void Padding_Conv2d_2c_3x3_Conv2D()
{
    printf("In Padding_Conv2d_2c_3x3_Conv2D \n");
    float input0[64*56*56];
    for (int i = 0; i < 64*56*56; i++){
        input0[i] = read_channel_intel(conv1_2b_out_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f);
        write_channel_intel(padding_2c_out_channel,result);
    }
    printf("Done Padding_Conv2d_2c_3x3_Conv2D \n");
}

__kernel void Conv2d_2c_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("In Conv2d_2c_3x3_Conv2D \n");
    float input0[215296];
    for (int i = 0; i < 215296; i++){
        input0[i] = read_channel_intel(padding_2c_out_channel);
    }
    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 56; ++yy)
        {
            for (int xx = 0; xx < 56; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 64; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 58) + yy) + ry) * 58) + xx) + rx)] * input1[((((((ff * 64) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_2c_out_channel, temp_0);
            }
        }
    }
    printf("Done Conv2d_2c_3x3_Conv2D \n");
}

__kernel void MaxPool_3a_3x3_MaxPool()
{
    printf("In MaxPool_3a_3x3_MaxPool \n");
    float input0[192*56*56];
    for (int i = 0; i < 192*56*56; i++){
        input0[i] = read_channel_intel(conv1_2c_out_channel);
    }
     //struct to store 256 bits of data
    struct concat_3a_buffer out;

    for (int ax1 = 0; ax1 < 192; ++ax1)
    {
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {
                float tensor = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((ax2 * 2) < (56 - rv)) && ((ax3 * 2) < (56 - rv1))) ? input0[((((((((ax1 * 28) + ax2) * 2) + rv) * 28) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
                out.concat_3a_out_buffer[((((ax1 * 28) + ax2) * 28) + ax3) % 8] = tensor;
                //After accumlating 256 bits, send the data through IO channel.
                if (((((ax1 * 28) + ax2) * 28) + ax3) % 8 == 7)
                {
                    write_channel_intel(concat_3a_out_channel, out);
                }   
            }
        } 
    }
    printf("Done MaxPool_3a_3x3_MaxPool \n");
}


/**
 * 2nd Inception module - Inception 3c
 */

//256 bits io channel struct
typedef struct concat_3b_buffer
    {
        float concat_3b_out_buffer[8];
    } concat_3b_struct;

// IO Channels for inception 3b to 3c
channel concat_3a_struct concat_3a_in_channel __attribute__((depth(10))) __attribute__((io("concat_3a"))); // Channel Tx
channel concat_3b_struct concat_3b_out_channel __attribute__((depth(10))) __attribute__((io("concat_3b"))); // Channel Rx


channel concat_3a_struct concat_3b_in_b0_channel __attribute__((depth(10))) ; // internal channel Branch 1
channel concat_3a_struct concat_3b_in_b1_channel __attribute__((depth(10))) ; // internal channel Branch 2
channel concat_3a_struct concat_3b_in_b2_channel __attribute__((depth(10))) ; // internal channel Branch 3
channel concat_3a_struct concat_3b_in_b3_channel __attribute__((depth(10))) ; // internal channel Branch 4

//branch 0
channel float conv1_3b_out_b0_channel __attribute__((depth(32)));
//branch 1
channel float conv2_1_3b_out_b1_channel __attribute__((depth(32)));
channel float padding_3b_out_b1_channel __attribute__((depth(32)));
channel float conv2_2_3b_out_b1_channel __attribute__((depth(32)));
//branch 2
channel float conv3_1_3b_out_b2_channel __attribute__((depth(32)));
channel float padding_3b_out_b2_channel __attribute__((depth(32)));
channel float conv3_2_3b_out_b2_channel __attribute__((depth(32)));
//branch 3
channel float maxpool_3b_out_b3_channel __attribute__((depth(32)));
channel float conv3_1_3b_out_b3_channel __attribute__((depth(32)));
//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_3b()
{
    printf(" In Feeder 3b");
    for (int i = 0; i < 18816; i++)
    {
        struct concat_3a_buffer input = read_channel_intel(concat_3a_in_channel);
        write_channel_intel(concat_3b_in_b0_channel, input);
        write_channel_intel(concat_3b_in_b1_channel, input);
        write_channel_intel(concat_3b_in_b2_channel, input);
        write_channel_intel(concat_3b_in_b3_channel, input);
    }
    printf(" Done Feeder 3b");
}


__kernel void Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("In Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D \n");
    float input0[192*28*28];
    for (int i = 0; i < 18816; i++){
        struct concat_3a_buffer in;
        in = read_channel_intel(concat_3b_in_b0_channel);
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_3a_out_buffer[k];
        }
    }
    printf("Done Reading from ChannelMixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D\n");
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    temp_0 += (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_3b_out_b0_channel, temp_0);
            }
        }
    }
    printf("Done Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("In Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D \n");
    float input0[192*28*28];
    for (int i = 0; i < 18816; i++){
        struct concat_3a_buffer in;
        in = read_channel_intel(concat_3b_in_b1_channel);
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_3a_out_buffer[k];
        }
    }
    printf("Done Reading from ChannelMixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D\n");
    for (int ff = 0; ff < 96; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    temp_0 += (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_3b_out_b1_channel, temp_0);
            }
        }
    }
    printf("Done Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Padding_Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    printf("In Padding_Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D \n");
    float input0[96*28*28];
    for (int i = 0; i < 96*28*28; i++){
        input0[i] = read_channel_intel(conv2_1_3b_out_b1_channel);
    }
     printf("Done Reading from ChannelPadding_Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D\n");
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 86400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
         float result = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
         write_channel_intel(padding_3b_out_b1_channel,result);
    }
    printf("Done Padding_Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("In Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D \n");
    float input0[86400];
    for (int i = 0; i < 86400; i++){
        input0[i] = read_channel_intel(padding_3b_out_b1_channel);
    }
    
    printf("Done Reading from ChannelMixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D\n");
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 96; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 96) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_3b_out_b1_channel, temp_0);
            }
        }
    }
    printf("Done Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    
    printf("In Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D \n");
    float input0[192*28*28];
    for (int i = 0; i < 18816; i++){
        struct concat_3a_buffer in;
        in = read_channel_intel(concat_3b_in_b2_channel);
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_3a_out_buffer[k];
        }
    }
    printf("Done Reading from ChannelMixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D\n");
    for (int ff = 0; ff < 16; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    temp_0 += (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_3b_out_b2_channel, temp_0);
            }
        }
    }
    printf("Done Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D \n");
}

__kernel void Padding_Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    printf("In Padding_Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D \n");
    float input0[16*28*28];
    for (int i = 0; i < 16*28*28; i++){
        input0[i] = read_channel_intel(conv3_1_3b_out_b2_channel);
    }
    printf("Done Reading from ChannelPadding_Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D\n");
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 14400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result  = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
        write_channel_intel(padding_3b_out_b2_channel,result);
    }
    printf("Done Padding_Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D \n");
}

__kernel void Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("In Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D \n");
    float input0[14400];
    for (int i = 0; i < 14400; i++){
        input0[i] = read_channel_intel(padding_3b_out_b2_channel);
    }
    printf("Done Reading from ChannelMixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D\n");
    for (int ff = 0; ff < 32; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 16; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 16) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv3_2_3b_out_b2_channel, temp_0);
            }
        }
    }
    printf("Done Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D \n");
}


__kernel void Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool()
{
    printf("In Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool \n");
    float input0[192*28*28];
    for (int i = 0; i < 18816; i++){
        struct concat_3a_buffer in;
        in = read_channel_intel(concat_3b_in_b3_channel);
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_3a_out_buffer[k];
        }
    }
    printf("Done Reading from ChannelMixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool\n");
    for (int ax1 = 0; ax1 < 192; ++ax1)
    {
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {
                float tensor = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? input0[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
                        
                    }
                }
                write_channel_intel(maxpool_3b_out_b3_channel, tensor);
            }
        }
    }
    printf("Done Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool \n");
}
__kernel void Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf("In Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D \n");
    float input0[192*28*28];
    for (int i = 0; i < 192*28*28; i++){
        input0[i] = read_channel_intel(maxpool_3b_out_b3_channel);
    }
    
     printf("Done Reading from ChannelMixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D\n");
    for (int ff = 0; ff < 32; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    temp_0 += (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_3b_out_b3_channel, temp_0);
            }
        }
    }
     printf("Done Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D \n");
}

__kernel void Mixed_3b_concat()
{
    printf("In Mixed_3b_concat \n");
    //struct to store 256 bits of data
    struct concat_3b_buffer out;

    float input0[64*28*28];
    for (int i = 0; i < 64*28*28; i++ ){
        input0[i] = read_channel_intel(conv1_3b_out_b0_channel);
    }
    float input1[128*28*28];
    for (int i = 0; i < 128*28*28; i++){
        input1[i] = read_channel_intel(conv2_2_3b_out_b1_channel);
    }
    float input2[32*28*28], input3[32*28*28] ;
    for (int i = 0; i < 32*28*28; i++){
        input2[i] = read_channel_intel(conv3_2_3b_out_b2_channel);
        input3[i] = read_channel_intel(conv3_1_3b_out_b3_channel);
    }
  

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result =(float)((175616 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -175616)] : (float)((150528 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -150528)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_3b_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //printf("\t 3b %d - %d --- %f \n",ax0_ax1_fused_ax2_fused_ax3_fused_inner,ax0_ax1_fused_ax2_fused_ax3_fused_inner%8,result);
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_3b_out_channel, out);
        }
    }
    printf("Done Mixed_3b_concat \n");
}

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

//internal channels from 3b to 3c
//branch 0
channel float conv1_3c_out_b0_channel __attribute__((depth(32)));
//branch 1
channel float conv2_1_3c_out_b1_channel __attribute__((depth(32)));
channel float padding_3c_out_b1_channel __attribute__((depth(32)));
channel float conv2_2_3c_out_b1_channel __attribute__((depth(32)));
//branch 2
channel float conv3_1_3c_out_b2_channel __attribute__((depth(32)));
channel float padding_3c_out_b2_channel __attribute__((depth(32)));
channel float conv3_2_3c_out_b2_channel __attribute__((depth(32)));
//branch 3
channel float maxpool_3c_out_b3_channel __attribute__((depth(32)));
channel float conv3_1_3c_out_b3_channel __attribute__((depth(32)));

//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_3c()
{
    printf(" In Feeder 3c\n\n \n ");
    for (int i = 0; i < 25088; i++)
    {
        struct concat_3b_buffer input = read_channel_intel(concat_3c_in_channel);
        write_channel_intel(concat_3c_in_b0_channel, input);
        write_channel_intel(concat_3c_in_b1_channel, input);
        write_channel_intel(concat_3c_in_b2_channel, input);
        write_channel_intel(concat_3c_in_b3_channel, input);
    }
    printf(" Done Feeder 3c\n\n \n ");
}

__kernel void Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf(" In Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D\n \n ");
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

                float temp_0 = input2[ff];
                for (int rc = 0; rc < 256; ++rc)
                {
                    temp_0 += (convInput[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_3c_out_b0_channel, temp_0);
            }
        }
    }
    printf(" Done  Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D\n \n ");
}

__kernel void Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)

{
    printf("  In Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D\n \n ");
    //Read Input from IO channel
    float convInput[200704];
    for (int i = 0; i < 25088; i++)
    {
        //struct to store 256 bits of data
        struct concat_3b_buffer in;
        in = read_channel_intel(concat_3c_in_b1_channel);
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
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 256; ++rc)
                {
                    temp_0 += (convInput[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_3c_out_b1_channel, temp_0);
            }
        }
    }
    printf("  Done Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D\n \n ");
}
__kernel void Padding_Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    printf(" In Padding_Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D\n \n ");
    float input0[128 * 28 * 28];
    for (int i = 0; i < 128 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(conv2_1_3c_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_3c_out_b1_channel, (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f));
    }
    printf(" Done Padding_Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D\n \n ");
}

__kernel void Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    printf(" In Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D\n ");
    float input0[115200];
    for (int i = 0; i < 115200; i++)
    {
        input0[i] = read_channel_intel(padding_3c_out_b1_channel);
    }

    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 128; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_3c_out_b1_channel, temp_0);
            }
        }
    }
    printf(" Done Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D\n ");
}

__kernel void Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    printf(" In Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D\n ");
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
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 256; ++rc)
                {
                    temp_0 += (convInput[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_3c_out_b2_channel, temp_0);
            }
        }
    }
    printf(" Done Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D\n ");
}

__kernel void Padding_Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    printf(" In Padding_Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D\n ");
    float input0[32 * 28 * 28];
    for (int i = 0; i < 32 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(conv3_1_3c_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28800; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_3c_out_b2_channel, (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f));
    }
    printf(" Done Padding_Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D\n ");
}

__kernel void Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    printf(" In Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D\n ");
    float input0[28800];
    for (int i = 0; i < 28800; i++)
    {
        input0[i] = read_channel_intel(padding_3c_out_b2_channel);
    }

    for (int ff = 0; ff < 96; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 32; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv3_2_3c_out_b2_channel, temp_0);
            }
        }
    }
    printf(" Done Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D\n ");
}

__kernel void Mixed_3c_Branch_3_MaxPool_0a_3x3_MaxPool()
{
    printf(" In Mixed_3c_Branch_3_MaxPool_0a_3x3_MaxPool\n ");
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

                float tensor = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? maxInput[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_3c_out_b3_channel, tensor);
            }
        }
    }
    printf(" Done Mixed_3c_Branch_3_MaxPool_0a_3x3_MaxPool\n ");
}

__kernel void Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    printf(" In Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D\n ");
    float input0[256 * 28 * 28];
    for (int i = 0; i < 256 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(maxpool_3c_out_b3_channel);
    }
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 28; ++yy)

        {
            for (int xx = 0; xx < 28; ++xx)

            {

                float temp_0 = input2[ff];

                for (int rc = 0; rc < 256; ++rc)
                {
                    temp_0 += (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_3c_out_b3_channel, temp_0);
            }
        }
    }
    printf(" Done Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D\n ");
}

__kernel void Mixed_3c_concat()
{
    //struct to store 256 bits of data
    struct concat_3c_buffer out;
    float input0[128 * 28 * 28];
    printf(" In  Mixed_3c_concat\n ");
    for (int i = 0; i < 128 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(conv1_3c_out_b0_channel);
    }
    float input1[192 * 28 * 28];
    for (int i = 0; i < 192 * 28 * 28; i++)
    {
        input1[i] = read_channel_intel(conv2_2_3c_out_b1_channel);
    }
    float input2[96 * 28 * 28];
    for (int i = 0; i < 96 * 28 * 28; i++)
    {
        input2[i] = read_channel_intel(conv3_2_3c_out_b2_channel);
    }
    float input3[64 * 28 * 28];
    for (int i = 0; i < 64 * 28 * 28; i++)
    {
        input3[i] = read_channel_intel(conv3_1_3c_out_b3_channel);
    }

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 376320; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        //T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((326144 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -326144)] : (float)((250880 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -250880)] : (float)((100352 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -100352)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        float result = (float)((326144 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -326144)] : (float)((250880 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -250880)] : (float)((100352 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -100352)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_3c_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //printf("\t 3c %d - %d --- %f \n", ax0_ax1_fused_ax2_fused_ax3_fused_inner, ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8, result);
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_3c_out_channel, out);
        }
    }
    printf(" Done  Mixed_3c_concat\n ");
}

typedef struct concat_4b_buffer {
    float concat_4b_out_buffer[8];
} concat_4b_struct;

// IO Channels for inception 3c to 4a
channel concat_3c_struct concat_4a_in_channel __attribute__((depth(10))) __attribute__((io("concat_3c"))); // Channel Rx
channel concat_4b_struct concat_4b_out_channel __attribute__((depth(10))) __attribute__((io("concat_4b"))); // Channel Tx

channel concat_3c_struct concat_4a_in_max_channel __attribute__((depth(10))) ; // internal channel maxpool

//internal channels
//branch 4a
channel float maxpool_4a_out_channel1 __attribute__((depth(32)));
channel float maxpool_4a_out_channel2 __attribute__((depth(32)));
channel float maxpool_4a_out_channel3 __attribute__((depth(32)));
channel float maxpool_4a_out_channel4 __attribute__((depth(32)));

//branch 0
channel float conv1_4b_out_b0_channel __attribute__((depth(32)));
//branch 1
channel float conv2_1_4b_out_b1_channel __attribute__((depth(32)));
channel float padding_4b_out_b1_channel __attribute__((depth(32)));
channel float conv2_2_4b_out_b1_channel __attribute__((depth(32)));

//branch 2
channel float conv3_1_4b_out_b2_channel __attribute__((depth(32)));
channel float padding_4b_out_b2_channel __attribute__((depth(32)));
channel float conv3_2_4b_out_b2_channel __attribute__((depth(32)));

//branch 3
channel float maxpool_4b_out_b3_channel __attribute__((depth(32)));
channel float conv4_1_4b_out_b3_channel __attribute__((depth(32)));

//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_4a()
{
    for (int i = 0; i < 47040; i++)
    {
        struct concat_3c_buffer input = read_channel_intel(concat_4a_in_channel);
        write_channel_intel(concat_4a_in_max_channel, input);
    }
}



__kernel void MaxPool_4a_3x3_MaxPool()
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
        }
    }

    for (int ax1 = 0; ax1 < 480; ++ax1)
    {
        for (int ax2 = 0; ax2 < 14; ++ax2)
        {
            for (int ax3 = 0; ax3 < 14; ++ax3)
            {
                float tensor = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
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
}

__kernel void Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[480*14*14];
    for (int i = 0; i < 480*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4a_out_channel1);
    }
    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 480; ++rc)
                {
                    temp_0 += (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_4b_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[480*14*14];
    for (int i = 0; i < 480*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4a_out_channel2);
    }

    for (int ff = 0; ff < 96; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 480; ++rc)
                {
                    temp_0 += (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_4b_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Padding_Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[96*14*14];
    for (int i = 0; i < 96*14*14; i++){
        input0[i] = read_channel_intel(conv2_1_4b_out_b1_channel);
    }
    
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 24576; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4b_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}

__kernel void Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[24576];
    for (int i = 0; i < 24576; i++){
        input0[i] = read_channel_intel(padding_4b_out_b1_channel);
    }
    

    for (int ff = 0; ff < 208; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 96; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 96) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_4b_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[480*14*14];
    for (int i = 0; i < 480*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4a_out_channel3);
    }

    for (int ff = 0; ff < 16; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 480; ++rc)
                {
                    temp_0 += (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_4b_out_b2_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[16*14*14];
    for (int i = 0; i < 16*14*14; i++){
        input0[i] = read_channel_intel(conv3_1_4b_out_b2_channel);
    }
    
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 4096; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4b_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[4096];
    for (int i = 0; i < 4096; i++){
        input0[i] = read_channel_intel(padding_4b_out_b2_channel);
    }
    for (int ff = 0; ff < 48; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 16; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 16) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv3_2_4b_out_b2_channel, temp_0);
            }
        }
    }
}


__kernel void Mixed_4b_Branch_3_MaxPool_0a_3x3_MaxPool()
{
    float input0[480*14*14];
    for (int i = 0; i < 480*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4a_out_channel4);
    }
    for (int ax1 = 0; ax1 < 480; ++ax1)
    {
        for (int ax2 = 0; ax2 < 14; ++ax2)
        {
            for (int ax3 = 0; ax3 < 14; ++ax3)
            {
                float tensor = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input0[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_4b_out_b3_channel, tensor);
            }
        }
    }
}

__kernel void Mixed_4b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[480*14*14];
    for (int i = 0; i < 480*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4b_out_b3_channel);
    }

    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 480; ++rc)
                {
                    temp_0 += (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv4_1_4b_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4b_concat()
{
    //struct to store 256 bits of data
    struct concat_4b_buffer out;

    float input0[192*14*14];
    for (int i = 0; i < 192*14*14; i++ ){
        input0[i] = read_channel_intel(conv1_4b_out_b0_channel);
    }
    float input1[208*14*14];
    for (int i = 0; i < 208*14*14; i++){
        input1[i] = read_channel_intel(conv2_2_4b_out_b1_channel);
    }
    float input2[48*14*14];
    for (int i = 0; i < 48*14*14; i++){
        input2[i] = read_channel_intel(conv3_2_4b_out_b2_channel);
    }
    float input3[64*14*14];
    for (int i = 0; i < 64*14*14; i++){
        input3[i] = read_channel_intel(conv4_1_4b_out_b3_channel);
    }

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_4b_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //printf("\t 4b %d - %d --- %f \n", ax0_ax1_fused_ax2_fused_ax3_fused_inner, ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8, result);
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_4b_out_channel, out);
        }

    }
}

typedef struct concat_4c_buffer {
    float concat_4c_out_buffer[8];
} concat_4c_struct;

// IO Channels for inception 4b to 4c
channel concat_4b_struct concat_4c_in_channel __attribute__((depth(10))) __attribute__((io("concat_4b"))); // Channel Rx
channel concat_4c_struct concat_4c_out_channel __attribute__((depth(10))) __attribute__((io("concat_4c"))); // Channel Tx

channel concat_4b_struct concat_4c_in_b0_channel __attribute__((depth(10))) ; // internal channel Branch 1
channel concat_4b_struct concat_4c_in_b1_channel __attribute__((depth(10))) ; // internal channel Branch 2
channel concat_4b_struct concat_4c_in_b2_channel __attribute__((depth(10))) ; // internal channel Branch 3
channel concat_4b_struct concat_4c_in_b3_channel __attribute__((depth(10))) ; // internal channel Branch 4


//internal channles
//branch 0
channel float conv1_4c_out_b0_channel __attribute__((depth(32)));

//branch 1
channel float conv2_1_4c_out_b1_channel __attribute__((depth(32)));
channel float padding_4c_out_b1_channel __attribute__((depth(32)));
channel float conv2_2_4c_out_b1_channel __attribute__((depth(32)));

//branch 2
channel float conv3_1_4c_out_b2_channel __attribute__((depth(32)));
channel float padding_4c_out_b2_channel __attribute__((depth(32)));
channel float conv3_2_4c_out_b2_channel __attribute__((depth(32)));

//branch 3
channel float maxpool_4c_out_b3_channel __attribute__((depth(32)));
channel float conv4_1_4c_out_b3_channel __attribute__((depth(32)));

//Feeder kernels to read data from IO and feed it into internal channnels
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


__kernel void Mixed_4c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
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
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_0 += (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv1_4c_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
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
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_0 += (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_4c_out_b1_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4c_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[112*14*14];
    for (int i = 0; i < 112*14*14; i++){
        input0[i] = read_channel_intel(conv2_1_4c_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28672; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4c_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[28672];
    for (int i = 0; i < 28672; i++){
        input0[i] = read_channel_intel(padding_4c_out_b1_channel);
    }
    for (int ff = 0; ff < 224; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 112; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 112) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_4c_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
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
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_0 += (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0: 0.000000e+00f;
                write_channel_intel(conv3_1_4c_out_b2_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4c_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[24*14*14];
    for (int i = 0; i < 24*14*14; i++){
        input0[i] = read_channel_intel(conv3_1_4c_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4c_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[6144];
    for (int i = 0; i < 6144; i++){
        input0[i] = read_channel_intel(padding_4c_out_b2_channel);
    }
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 24; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 24) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv3_2_4c_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4c_Branch_3_MaxPool_0a_3x3_MaxPool()
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
                float tensor = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? maxInput[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_4c_out_b3_channel, tensor);
            }
        }
    }
}

__kernel void Mixed_4c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[512*14*14];
    for (int i = 0; i < 512*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4c_out_b3_channel);
    }
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_0 += (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv4_1_4c_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4c_concat()
{
    //struct to store 256 bits of data
    struct concat_4c_buffer out;

    float input0[160*14*14];
    for (int i = 0; i < 160*14*14; i++ ){
        input0[i] = read_channel_intel(conv1_4c_out_b0_channel);
    }
    float input1[224*14*14];
    for (int i = 0; i < 224*14*14; i++){
        input1[i] = read_channel_intel(conv2_2_4c_out_b1_channel);
    }
    float input2[64*14*14], input3[64*14*14];
    for (int i = 0; i < 64*14*14; i++){
        input2[i] = read_channel_intel(conv3_2_4c_out_b2_channel);
        input3[i] = read_channel_intel(conv4_1_4c_out_b3_channel);
    }
    

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((75264 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -75264)] : (float)((31360 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -31360)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_4c_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //printf("\t 4c %d - %d --- %f \n", ax0_ax1_fused_ax2_fused_ax3_fused_inner, ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8, result);
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_4c_out_channel, out);
        }

    }
}