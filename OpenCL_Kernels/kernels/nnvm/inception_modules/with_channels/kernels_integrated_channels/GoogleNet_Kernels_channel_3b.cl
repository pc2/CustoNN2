//enable channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable
//256 bits io channel struct
typedef struct concat_3b_buffer
    {
        float concat_3b_out_buffer[8];
    } concat_3b_struct;
// IO Channels for inception 3b to 3c
channel concat_3b_struct concat_3b_out_channel __attribute__((depth(10))) __attribute__((io("kernel_output_ch0"))); // Channel Tx

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
//branch 3a
channel float maxpool1_3a_out_channel __attribute__((depth(32)));
channel float maxpool2_3a_out_channel __attribute__((depth(32)));
channel float maxpool3_3a_out_channel __attribute__((depth(32)));
channel float maxpool4_3a_out_channel __attribute__((depth(32)));
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







__kernel void Padding_Conv2d_1a_7x7_Conv2D(__global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 157323; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_1a_out_channel, (float)(((((458 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) < 51754)) && (2 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229) < 226)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) / 229) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52441)) + -1350)] : 0.000000e+00f));
    }
}

__kernel void Conv2d_1a_7x7_Conv2D(__global float *restrict input1, __global float *restrict input2)


    

{
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
}

/*__kernel void Padding_MaxPool_2a_3x3_MaxPool(__global float *restrict T_transpose, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner]), 0.000000e+00f);
    }
}*/

__kernel void MaxPool_2a_3x3_MaxPool()
{
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
}

__kernel void Conv2d_2b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[64*56*56];
    for (int i = 0; i < 64*56*56; i++){
        input0[i] = read_channel_intel(maxpool_2a_out_channel);
    }
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
}

__kernel void Padding_Conv2d_2c_3x3_Conv2D()
{
    float input0[64*56*56];
    for (int i = 0; i < 64*56*56; i++){
        input0[i] = read_channel_intel(conv1_2b_out_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_2c_out_channel, (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f));
    }
}

__kernel void Conv2d_2c_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
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
}

/*__kernel void Padding_MaxPool_3a_3x3_MaxPool(__global float *restrict T_transpose, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 192) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))];
    }
}*/

__kernel void MaxPool_3a_3x3_MaxPool()
{
    float input0[192*56*56];
    for (int i = 0; i < 192*56*56; i++){
        input0[i] = read_channel_intel(conv1_2c_out_channel);
    }
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
                write_channel_intel(maxpool1_3a_out_channel, tensor);
                write_channel_intel(maxpool2_3a_out_channel, tensor);
                write_channel_intel(maxpool3_3a_out_channel, tensor);
                write_channel_intel(maxpool4_3a_out_channel, tensor);
            }
        }
    }
}
/*__kernel void MaxPool_3a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
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
                write_channel_intel(maxpool1_3a_out_channel, tensor[((((ax1 * 28) + ax2) * 28) + ax3)]);
                write_channel_intel(maxpool2_3a_out_channel, tensor[((((ax1 * 28) + ax2) * 28) + ax3)]);
                write_channel_intel(maxpool3_3a_out_channel, tensor[((((ax1 * 28) + ax2) * 28) + ax3)]);
                write_channel_intel(maxpool4_3a_out_channel, tensor[((((ax1 * 28) + ax2) * 28) + ax3)]);
            }
        }
    }
}
*/
/*__kernel void fuse_transpose_48_kernel0(__global float *restrict T_transpose, __global float *restrict input0)
{

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 192) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 192))];
    }
}*/

/*__kernel void Padding_Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 192) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))];
    }
}*/

__kernel void Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[192*28*28];
    for (int i = 0; i < 192*28*28; i++){
        input0[i] = read_channel_intel(maxpool1_3a_out_channel);
    }
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
}

__kernel void Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[192*28*28];
    for (int i = 0; i < 192*28*28; i++){
        input0[i] = read_channel_intel(maxpool2_3a_out_channel);
    }
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
}

__kernel void Padding_Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[96*28*28];
    for (int i = 0; i < 96*28*28; i++){
        input0[i] = read_channel_intel(conv2_1_3b_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 86400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_3b_out_b1_channel, (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f));
    }
}

__kernel void Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[86400];
    for (int i = 0; i < 86400; i++){
        input0[i] = read_channel_intel(padding_3b_out_b1_channel);
    }
    
    
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
}

__kernel void Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[192*28*28];
    for (int i = 0; i < 192*28*28; i++){
        input0[i] = read_channel_intel(maxpool3_3a_out_channel);
    }
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
}

__kernel void Padding_Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[16*28*28];
    for (int i = 0; i < 16*28*28; i++){
        input0[i] = read_channel_intel(conv3_1_3b_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 14400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_3b_out_b2_channel, (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f));
    }
}

__kernel void Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[14400];
    for (int i = 0; i < 14400; i++){
        input0[i] = read_channel_intel(padding_3b_out_b2_channel);
    }
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
}

/*__kernel void Padding_Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict T_transpose, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 192) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))];
    }
}*/

__kernel void Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool()
{
    float input0[192*28*28];
    for (int i = 0; i < 192*28*28; i++){
        input0[i] = read_channel_intel(maxpool4_3a_out_channel);
    }
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
}
__kernel void Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[192*28*28];
    for (int i = 0; i < 192*28*28; i++){
        input0[i] = read_channel_intel(maxpool_3b_out_b3_channel);
    }
    

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
}

__kernel void Mixed_3b_concat(__global float *restrict T_concat)
{
    //struct to store 256 bits of data
    struct concat_3b_buffer out;

    /*float input0[64*28*28];
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
    }*/
  

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
      /*  T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((175616 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -175616)] : (float)((150528 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -150528)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));*/
        T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((175616 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? read_channel_intel(conv3_1_3b_out_b3_channel) : (float)((150528 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? read_channel_intel(conv3_2_3b_out_b2_channel) : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? read_channel_intel(conv2_2_3b_out_b1_channel) : read_channel_intel(conv1_3b_out_b0_channel))));
        out.concat_3b_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
        
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_3b_out_channel, out);
        }
        
        
    }
}
