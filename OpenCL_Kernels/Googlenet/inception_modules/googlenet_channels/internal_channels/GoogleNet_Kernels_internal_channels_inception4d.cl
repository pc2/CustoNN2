//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//branch 0
channel float conv1_4d_out_b0_channel __attribute__((depth(32)));
//branch 1
channel float conv2_1_4d_out_b1_channel __attribute__((depth(32)));
channel float padding_4d_out_b1_channel __attribute__((depth(32)));
channel float conv2_2_4d_out_b1_channel __attribute__((depth(32)));
//branch 2
channel float conv3_1_4d_out_b2_channel __attribute__((depth(32)));
channel float padding_4d_out_b2_channel __attribute__((depth(32)));
channel float conv3_2_4d_out_b2_channel __attribute__((depth(32)));
//branch 3
channel float maxpool_4d_out_b3_channel __attribute__((depth(32)));
channel float conv4_1_4d_out_b3_channel __attribute__((depth(32)));






__kernel void Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    for (int ff = 0; ff < 128; ++ff)
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
                write_channel_intel(conv1_4d_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4d_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    for (int ff = 0; ff < 128; ++ff)
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
                write_channel_intel(conv2_1_4d_out_b1_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[128*14*14];
    for (int i = 0; i < 128*14*14; i++){
        input0[i] = read_channel_intel(conv2_1_4d_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 32768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
       write_channel_intel(padding_4d_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[32768];
    for(int i = 0; i < 32768; i++){
        input0[i] = read_channel_intel(padding_4d_out_b1_channel);
    }
    for (int ff = 0; ff < 256; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 128; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_4d_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4d_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    for (int ff = 0; ff < 24; ++ff)
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
                write_channel_intel(conv3_1_4d_out_b2_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[24*14*14];
    for(int i = 0; i < 24*14*14; i++){
        input0[i] = read_channel_intel(conv3_1_4d_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4d_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[6144];
    for(int i = 0; i < 6144; i++){
        input0[i] = read_channel_intel(padding_4d_out_b2_channel);
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
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_2_4d_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4d_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict input0)
{
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
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input0[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_4d_out_b3_channel, tensor);
            }
        }
    }
}

__kernel void Mixed_4d_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[512*14*14];
    for(int i = 0; i < 512*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4d_out_b3_channel);
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
                write_channel_intel(conv4_1_4d_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4d_concat()
{
    float input0[128*14*14];
    for (int i = 0; i < 128*14*14; i++ ){
        input0[i] = read_channel_intel(conv1_4d_out_b0_channel);
    }
    float input1[256*14*14];
    for (int i = 0; i < 256*14*14; i++){
        input1[i] = read_channel_intel(conv2_2_4d_out_b1_channel);
    }
    float input2[64*14*14], input3[64*14*14];
    for (int i = 0; i < 64*14*14; i++){
        input2[i] = read_channel_intel(conv3_2_4d_out_b2_channel);
        input3[i] = read_channel_intel(conv4_1_4d_out_b3_channel);
    }
    

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] : (float)((75264 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -75264)] : (float)((25088 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -25088)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}
