#pragma OPENCL EXTENSION cl_intel_channels : enable
//256 bits io channel struct
typedef struct concat_3b_buffer
{
    float concat_3b_out_buffer[8];
} concat_3b_struct;

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


typedef struct concat_4d_buffer {
    float concat_4d_out_buffer[8];
} concat_4d_struct;

// IO Channels for inception 4c to 4d
channel concat_4c_struct concat_4d_in_channel __attribute__((depth(10))) __attribute__((io("concat_4c"))); // Channel Rx
channel concat_4d_struct concat_4d_out_channel __attribute__((depth(10))) __attribute__((io("concat_4d"))); // Channel Tx

channel concat_4c_struct concat_4d_in_b0_channel __attribute__((depth(10))) ; // internal channel Branch 1
channel concat_4c_struct concat_4d_in_b1_channel __attribute__((depth(10))) ; // internal channel Branch 2
channel concat_4c_struct concat_4d_in_b2_channel __attribute__((depth(10))) ; // internal channel Branch 3
channel concat_4c_struct concat_4d_in_b3_channel __attribute__((depth(10))) ; // internal channel Branch 4


//internal channels
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




//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_4d()
{
    for (int i = 0; i < 12544; i++)
    {
        struct concat_4c_buffer input = read_channel_intel(concat_4d_in_channel);
        write_channel_intel(concat_4d_in_b0_channel, input);
        write_channel_intel(concat_4d_in_b1_channel, input);
        write_channel_intel(concat_4d_in_b2_channel, input);
        write_channel_intel(concat_4d_in_b3_channel, input);
    }
}


__kernel void Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4c_buffer in;
        in = read_channel_intel(concat_4d_in_b0_channel);
        
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4c_out_buffer[k];
        }
    }

    for (int ff = 0; ff < 128; ++ff)
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
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_4d_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4d_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4c_buffer in;
        in = read_channel_intel(concat_4d_in_b1_channel);
        
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4c_out_buffer[k];
        }
    }

    for (int ff = 0; ff < 128; ++ff)
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

__kernel void Mixed_4d_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4c_buffer in;
        in = read_channel_intel(concat_4d_in_b2_channel);
        
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4c_out_buffer[k];
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

__kernel void Mixed_4d_Branch_3_MaxPool_0a_3x3_MaxPool()
{
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
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? maxInput[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
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
    //struct to store 256 bits of data
    struct concat_4d_buffer out;
    

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
        out.concat_4d_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //printf("\t 4d %d - %d --- %f \n", ax0_ax1_fused_ax2_fused_ax3_fused_inner, ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8, result);
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_4d_out_channel, out);
        }

    }
}

typedef struct concat_4e_buffer {
    float concat_4e_out_buffer[8];
} concat_4e_struct;

// IO Channels for inception 4d to 4e
channel concat_4d_struct concat_4e_in_channel __attribute__((depth(10))) __attribute__((io("concat_4d"))); // Channel Rx
channel concat_4e_struct concat_4e_out_channel __attribute__((depth(10))) __attribute__((io("concat_4e"))); // Channel Tx

channel concat_4d_struct concat_4e_in_b0_channel __attribute__((depth(10))) ; // internal channel Branch 1
channel concat_4d_struct concat_4e_in_b1_channel __attribute__((depth(10))) ; // internal channel Branch 2
channel concat_4d_struct concat_4e_in_b2_channel __attribute__((depth(10))) ; // internal channel Branch 3
channel concat_4d_struct concat_4e_in_b3_channel __attribute__((depth(10))) ; // internal channel Branch 4


//internal channels
//branch 0
channel float conv1_4e_out_b0_channel __attribute__((depth(32)));
//branch 1
channel float conv2_1_4e_out_b1_channel __attribute__((depth(32)));
channel float padding_4e_out_b1_channel __attribute__((depth(32)));
channel float conv2_2_4e_out_b1_channel __attribute__((depth(32)));
//branch 2
channel float conv3_1_4e_out_b2_channel __attribute__((depth(32)));
channel float padding_4e_out_b2_channel __attribute__((depth(32)));
channel float conv3_2_4e_out_b2_channel __attribute__((depth(32)));
//branch 3
channel float maxpool_4e_out_b3_channel __attribute__((depth(32)));
channel float conv4_1_4e_out_b3_channel __attribute__((depth(32)));




//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_4e()
{
    for (int i = 0; i < 12544; i++)
    {
        struct concat_4d_buffer input = read_channel_intel(concat_4e_in_channel);
        write_channel_intel(concat_4e_in_b0_channel, input);
        write_channel_intel(concat_4e_in_b1_channel, input);
        write_channel_intel(concat_4e_in_b2_channel, input);
        write_channel_intel(concat_4e_in_b3_channel, input);
    }
}


__kernel void Mixed_4e_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    
    //Read Input from IO channel
    float convInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b0_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4d_out_buffer[k];
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
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_4e_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4e_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b1_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4d_out_buffer[k];
        }
    }
    
    for (int ff = 0; ff < 144; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_0  += (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_4e_out_b1_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[144*14*14];
    for (int i = 0; i < 144*14*14; i++){
        input0[i] = read_channel_intel(conv2_1_4e_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 36864; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4e_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[36864];
    for (int i = 0; i < 36864; i++){
        input0[i] = read_channel_intel(padding_4e_out_b1_channel);
    }
    for (int ff = 0; ff < 288; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 144; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 144) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_4e_out_b1_channel, temp_0);
                
            }
        }
    }
}

__kernel void Mixed_4e_Branch_2_Conv2d_0a_1x1_Conv2D(
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b2_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4d_out_buffer[k];
        }
    }
    for (int ff = 0; ff < 32; ++ff)
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
                write_channel_intel(conv3_1_4e_out_b2_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[32*14*14];
    for (int i = 0; i < 32*14*14; i++){
        input0[i] = read_channel_intel(conv3_1_4e_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4e_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D(
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[8192];
    for (int i = 0; i < 8192; i++){
        input0[i] = read_channel_intel(padding_4e_out_b2_channel);
    }
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 32; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv3_2_4e_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4e_Branch_3_MaxPool_0a_3x3_MaxPool()
{
    //Read Input from IO channel
    float maxInput[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b3_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            maxInput[(i * 8) + k] = in.concat_4d_out_buffer[k];
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
                write_channel_intel(maxpool_4e_out_b3_channel, tensor);
            }
        }
    }
}

__kernel void Mixed_4e_Branch_3_Conv2d_0b_1x1_Conv2D(
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[512*14*14];
    for (int i = 0; i < 512*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4e_out_b3_channel);
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
                write_channel_intel(conv4_1_4e_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4e_concat()
{
    //struct to store 256 bits of data
    struct concat_4e_buffer out;
    
    float input0[112*14*14];
    for (int i = 0; i < 112*14*14; i++ ){
        input0[i] = read_channel_intel(conv1_4e_out_b0_channel);
    }
    float input1[288*14*14];
    for (int i = 0; i < 288*14*14; i++){
        input1[i] = read_channel_intel(conv2_2_4e_out_b1_channel);
    }
    float input2[64*14*14], input3[64*14*14];
    for (int i = 0; i < 64*14*14; i++){
        input2[i] = read_channel_intel(conv3_2_4e_out_b2_channel);
        input3[i] = read_channel_intel(conv4_1_4e_out_b3_channel);
    }
    
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((90944 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -90944)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((21952 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -21952)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_4e_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //printf("\t 4e %d - %d --- %f \n", ax0_ax1_fused_ax2_fused_ax3_fused_inner, ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8, result);
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_4e_out_channel, out);
        }

    }
}

typedef struct concat_4f_buffer {
    float concat_4f_out_buffer[8];
} concat_4f_struct;

// IO Channels for inception 4e to 4f
channel concat_4e_struct concat_4f_in_channel __attribute__((depth(10))) __attribute__((io("concat_4e"))); // Channel Rx
channel concat_4f_struct concat_4f_out_channel __attribute__((depth(10))) __attribute__((io("concat_4f"))); // Channel Tx

channel concat_4e_struct concat_4f_in_b0_channel __attribute__((depth(10))) ; // internal channel Branch 1
channel concat_4e_struct concat_4f_in_b1_channel __attribute__((depth(10))) ; // internal channel Branch 2
channel concat_4e_struct concat_4f_in_b2_channel __attribute__((depth(10))) ; // internal channel Branch 3
channel concat_4e_struct concat_4f_in_b3_channel __attribute__((depth(10))) ; // internal channel Branch 4


//internal channles
//branch 0
channel float conv1_4f_out_b0_channel __attribute__((depth(32)));

//branch 1
channel float conv2_1_4f_out_b1_channel __attribute__((depth(32)));
channel float padding_4f_out_b1_channel __attribute__((depth(32)));
channel float conv2_2_4f_out_b1_channel __attribute__((depth(32)));

//branch 2
channel float conv3_1_4f_out_b2_channel __attribute__((depth(32)));
channel float padding_4f_out_b2_channel __attribute__((depth(32)));
channel float conv3_2_4f_out_b2_channel __attribute__((depth(32)));

//branch 3
channel float maxpool_4f_out_b3_channel __attribute__((depth(32)));
channel float conv4_1_4f_out_b3_channel __attribute__((depth(32)));

//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_4f()
{
    for (int i = 0; i < 12936; i++)
    {
        struct concat_4e_buffer input = read_channel_intel(concat_4f_in_channel);
        write_channel_intel(concat_4f_in_b0_channel, input);
        write_channel_intel(concat_4f_in_b1_channel, input);
        write_channel_intel(concat_4f_in_b2_channel, input);
        write_channel_intel(concat_4f_in_b3_channel, input);
    }
}

__kernel void Mixed_4f_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[103488];
    // 103488/8 = 12936
    for (int i = 0; i < 12936; i++)
    {
        //struct to store 256 bits of data
        struct concat_4e_buffer in;
        in = read_channel_intel(concat_4f_in_b0_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4e_out_buffer[k];
        }
    }
    
    for (int ff = 0; ff < 256; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 528; ++rc)
                {
                    temp_0 += (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 528) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_4f_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4f_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[103488];
    // 103488/8 = 12936
    for (int i = 0; i < 12936; i++)
    {
        //struct to store 256 bits of data
        struct concat_4e_buffer in;
        in = read_channel_intel(concat_4f_in_b1_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4e_out_buffer[k];
        }
    }
    
    for (int ff = 0; ff < 160; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 528; ++rc)
                {
                    temp_0 += (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 528) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_4f_out_b1_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4f_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[160*14*14];
    for (int i = 0; i < 160*14*14; i++){
        input0[i] = read_channel_intel(conv2_1_4f_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40960; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4f_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4f_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[40960];
    for (int i = 0; i < 40960; i++){
        input0[i] = read_channel_intel(padding_4f_out_b1_channel);
    }
    for (int ff = 0; ff < 320; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 160; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 160) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_4f_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4f_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel
    float convInput[103488];
    // 103488/8 = 12936

    for (int i = 0; i < 12936; i++)
    {
        //struct to store 256 bits of data
        struct concat_4e_buffer in;
        in = read_channel_intel(concat_4f_in_b2_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            convInput[(i * 8) + k] = in.concat_4e_out_buffer[k];
        }
    }
    
    for (int ff = 0; ff < 32; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 528; ++rc)
                {
                    temp_0 += (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 528) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0: 0.000000e+00f;
                write_channel_intel(conv3_1_4f_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Padding_Mixed_4f_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[32*14*14];
    for (int i = 0; i < 32*14*14; i++){
        input0[i] = read_channel_intel(conv3_1_4f_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
         write_channel_intel(padding_4f_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4f_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[8192];
    for (int i = 0; i < 8192; i++){
        input0[i] = read_channel_intel(padding_4f_out_b2_channel);
    }
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 32; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_2_4f_out_b2_channel, temp_0);
                
            }
        }
    }
}


__kernel void Mixed_4f_Branch_3_MaxPool_0a_3x3_MaxPool()
{
    //Read Input from IO channel
    float maxInput[103488];
    // 103488/8 = 12936

    for (int i = 0; i < 12936; i++)
    {
        //struct to store 256 bits of data
        struct concat_4e_buffer in;
        in = read_channel_intel(concat_4f_in_b3_channel);
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            maxInput[(i * 8) + k] = in.concat_4e_out_buffer[k];
        }
    }
    
    for (int ax1 = 0; ax1 < 528; ++ax1)
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
                write_channel_intel(maxpool_4f_out_b3_channel, tensor);
            }
        }
    }
}

__kernel void Mixed_4f_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[528*14*14];
    for (int i = 0; i < 528*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4f_out_b3_channel);
    }
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 528; ++rc)
                {
                    temp_0 += (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 528) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv4_1_4f_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4f_concat()
{
    //struct to store 256 bits of data
    struct concat_4f_buffer out;
    
    float input0[256*14*14];
    for (int i = 0; i < 256*14*14; i++ ){
        input0[i] = read_channel_intel(conv1_4f_out_b0_channel);
    }
    float input1[320*14*14];
    for (int i = 0; i < 320*14*14; i++){
        input1[i] = read_channel_intel(conv2_2_4f_out_b1_channel);
    }
    float input2[128*14*14], input3[128*14*14];
    for (int i = 0; i < 128*14*14; i++){
        input2[i] = read_channel_intel(conv3_2_4f_out_b2_channel);
        input3[i] = read_channel_intel(conv4_1_4f_out_b3_channel);
    }
    
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 163072; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((137984 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -137984)] : (float)((112896 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -112896)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_4f_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //printf("\t 4f %d - %d --- %f \n", ax0_ax1_fused_ax2_fused_ax3_fused_inner, ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8, result);
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_4f_out_channel, out);
        }
    }
}

typedef struct concat_5b_buffer {
    float concat_5b_out_buffer[8];
} concat_5b_struct;

// IO Channels for inception 4f to 5a
channel concat_4f_struct concat_5a_in_channel __attribute__((depth(10))) __attribute__((io("concat_4f"))); // Channel Rx
channel concat_5b_struct concat_5b_out_channel __attribute__((depth(10))) __attribute__((io("concat_5b"))); // Channel Tx

channel concat_4f_struct concat_5a_in_max_channel __attribute__((depth(10))) ; // internal channel maxpool

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

channel concat_4f_struct concat_5a_in_b0_channel __attribute__((depth(10))) ; // internal channel maxpool
//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_5a()
{
    for (int i = 0; i < 20384; i++)
    {
        struct concat_4f_buffer input = read_channel_intel(concat_5a_in_channel);
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
    for (int ax1 = 0; ax1 < 832; ++ax1)
    {
        for (int ax2 = 0; ax2 < 7; ++ax2)
        {
            for (int ax3 = 0; ax3 < 7; ++ax3)
            {
                float tensor = -3.402823e+38f;
                for (int rv = 0; rv < 2; ++rv)
                {
                    for (int rv1 = 0; rv1 < 2; ++rv1)
                    {
                        tensor = max(tensor, maxInput[((((((((ax1 * 7) + ax2) * 2) + rv) * 7) + ax3) * 2) + rv1)]);
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
    float input0[832*7*7];
    for (int i = 0; i < 832*7*7; i++){
        input0[i] = read_channel_intel(maxpool_5a_out_channel1);
    }
    for (int ff = 0; ff < 256; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_0 += (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_5b_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{
    float input0[832*7*7];
    for (int i = 0; i < 832*7*7; i++){
        input0[i] = read_channel_intel(maxpool_5a_out_channel2);
    }
    for (int ff = 0; ff < 160; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0= input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_0 += (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_5b_out_b1_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[160*7*7];
    for (int i = 0; i < 160*7*7; i++){
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
    for (int i = 0; i < 12960; i++){
        input0[i] = read_channel_intel(padding_5b_out_b1_channel);
    }
    
    for (int ff = 0; ff < 320; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 160; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input1[((((((ff * 160) + rc) * 3) + ry) * 3) + rx)]);
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
    float input0[832*7*7];
    for (int i = 0; i < 832*7*7; i++){
        input0[i] = read_channel_intel(maxpool_5a_out_channel3);
    }
    for (int ff = 0; ff < 32; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_0 += (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_5b_out_b2_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D()
{
    float input0[32*7*7];
    for (int i = 0; i < 32*7*7; i++){
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
    for (int i = 0; i < 2592; i++){
        input0[i] = read_channel_intel(padding_5b_out_b2_channel);
    }
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 32; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]);
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
    float input0[832*7*7];
    for (int i = 0; i < 832*7*7; i++){
        input0[i] = read_channel_intel(maxpool_5a_out_channel4);
    }
    for (int ax1 = 0; ax1 < 832; ++ax1)
    {
        for (int ax2 = 0; ax2 < 7; ++ax2)
        {
            for (int ax3 = 0; ax3 < 7; ++ax3)
            {
                float tensor= -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (8 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (8 - rv1))) ? input0[(((((((ax1 * 7) + ax2) + rv) * 7) + ax3) + rv1) + -8)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_5b_out_b3_channel, tensor);
            }
        }
    }
}

__kernel void Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[832*7*7];
    for (int i = 0; i < 832*7*7; i++){
        input0[i] = read_channel_intel(maxpool_5b_out_b3_channel);
    }
    
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_0 += (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv4_1_5b_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5b_concat()
{
    //struct to store 256 bits of data
    struct concat_5b_buffer out;
    
    float input0[256*7*7];
    for (int i = 0; i < 256*7*7; i++ ){
        input0[i] = read_channel_intel(conv1_5b_out_b0_channel);
    }
    float input1[320*7*7];
    for (int i = 0; i < 320*7*7; i++){
        input1[i] = read_channel_intel(conv2_2_5b_out_b1_channel);
    }
    float input2[128*7*7];
    for (int i = 0; i < 128*7*7; i++){
        input2[i] = read_channel_intel(conv3_2_5b_out_b2_channel);
    }
    float input3[128*7*7];
    for (int i = 0; i < 128*7*7; i++){
        input3[i] = read_channel_intel(conv4_1_5b_out_b3_channel);
    }
    
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((34496 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -34496)] : (float)((28224 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -28224)] : (float)((12544 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -12544)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_5b_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        printf("\t 5b %d - %d --- %f \n", ax0_ax1_fused_ax2_fused_ax3_fused_inner, ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8, result);
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            write_channel_intel(concat_5b_out_channel, out);
        }
    }
}


// IO Channels for inception 5b to 5c
channel concat_5b_struct concat_5c_in_channel __attribute__((depth(10))) __attribute__((io("concat_5b"))); // Channel Rx

channel concat_5b_struct concat_5c_in_b0_channel __attribute__((depth(10))) ; // internal channel Branch 1
channel concat_5b_struct concat_5c_in_b1_channel __attribute__((depth(10))) ; // internal channel Branch 2
channel concat_5b_struct concat_5c_in_b2_channel __attribute__((depth(10))) ; // internal channel Branch 3
channel concat_5b_struct concat_5c_in_b3_channel __attribute__((depth(10))) ; // internal channel Branch 4

//internal channles
//branch 0
channel float conv1_5c_out_b0_channel __attribute__((depth(32)));

//branch 1
channel float conv2_1_5c_out_b1_channel __attribute__((depth(32)));
channel float padding_5c_out_b1_channel __attribute__((depth(32)));
channel float conv2_2_5c_out_b1_channel __attribute__((depth(32)));

//branch 2
channel float conv3_1_5c_out_b2_channel __attribute__((depth(32)));
channel float padding_5c_out_b2_channel __attribute__((depth(32)));
channel float conv3_2_5c_out_b2_channel __attribute__((depth(32)));

//branch 3
channel float maxpool_5c_out_b3_channel __attribute__((depth(32)));
channel float conv4_1_5c_out_b3_channel __attribute__((depth(32)));

//concat
channel float concat_5c_out_channel __attribute__((depth(32)));

//avgpool
channel float avgpool_out_channel __attribute__((depth(32)));

//final conv
//channel float conv1_0c_out_channel __attribute__((depth(32)));

//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_5c()
{
    for (int i = 0; i < 5096; i++)
    {
        struct concat_5b_buffer input = read_channel_intel(concat_5c_in_channel);
        write_channel_intel(concat_5c_in_b0_channel, input);
        write_channel_intel(concat_5c_in_b1_channel, input);
        write_channel_intel(concat_5c_in_b2_channel, input);
        write_channel_intel(concat_5c_in_b3_channel, input);
    }
}


__kernel void Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D( __global float *restrict input1, __global float *restrict input2)
{
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
    for (int ff = 0; ff < 384; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_0 += (convInput[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_5c_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
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
    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_0 += (convInput[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? + temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_5c_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Padding_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[192*7*7];
    for (int i = 0; i < 192*7*7; i++){
        input0[i] = read_channel_intel(conv2_1_5c_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 15552; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_5c_out_b1_channel, (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f));
    }
}
__kernel void Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[15552];
    for (int i = 0; i < 15552; i++){
        input0[i] = read_channel_intel(padding_5c_out_b1_channel);
    }
    for (int ff = 0; ff < 384; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input1[((((((ff * 192) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_5c_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
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
    
    for (int ff = 0; ff < 48; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_0 += (convInput[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0: 0.000000e+00f;
                write_channel_intel(conv3_1_5c_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Padding_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[48*7*7];
    for (int i = 0; i < 48*7*7; i++){
        input0[i] = read_channel_intel(conv3_1_5c_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 3888; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_5c_out_b2_channel, (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f));
    }
}
__kernel void Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[3888];
    for (int i = 0; i < 3888; i++){
        input0[i] = read_channel_intel(padding_5c_out_b2_channel);
    }
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 48; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_0 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input1[((((((ff * 48) + rc) * 3) + ry) * 3) + rx)]);
                        }
                    }
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv3_2_5c_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5c_Branch_3_MaxPool_0a_3x3_MaxPool()
{
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
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((((1 - rv) <= ax2) && (ax2 < (8 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (8 - rv1))) ? maxInput[(((((((ax1 * 7) + ax2) + rv) * 7) + ax3) + rv1) + -8)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_5c_out_b3_channel, tensor);
            }
        }
    }
}

__kernel void Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[832*7*7];
    for (int i = 0; i < 832*7*7; i++){
        input0[i] = read_channel_intel(maxpool_5c_out_b3_channel);
    }
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_0 += (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]);
                }
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv4_1_5c_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_5c_concat()
{
    //struct to store 256 bits of data
    //struct concat_5c_buffer out;
    
    float input0[384*7*7];
    for (int i = 0; i < 384*7*7; i++ ){
        input0[i] = read_channel_intel(conv1_5c_out_b0_channel);
    }
    float input1[384*7*7];
    for (int i = 0; i < 384*7*7; i++){
        input1[i] = read_channel_intel(conv2_2_5c_out_b1_channel);
    }
    float input2[128*7*7], input3[128*7*7];
    for (int i = 0; i < 128*7*7; i++){
        input2[i] = read_channel_intel(conv3_2_5c_out_b2_channel);
        input3[i] = read_channel_intel(conv4_1_5c_out_b3_channel);
    }
    
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((43904 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -43904)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : (float)((18816 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -18816)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        //out.concat_5c_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        //if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        //{
            write_channel_intel(concat_5c_out_channel, result);
        //}
         printf("\t 5c %d --- %f \n", ax0_ax1_fused_ax2_fused_ax3_fused_inner, result);
    }
}

__kernel void AvgPool_0a_7x7_AvgPool()
{
    float input0[50176];
    for (int i = 0; i < 50176; i++){
        input0[i] = read_channel_intel(concat_5c_out_channel);
    }
    for (int ax1 = 0; ax1 < 1024; ++ax1)
    {
        float tensor= 0.000000e+00f;
        for (int rv = 0; rv < 7; ++rv) 
        {
            for (int rv1 = 0; rv1 < 7; ++rv1)
            {
                tensor = (tensor + (input0[((((ax1 * 7) + rv) * 7) + rv1)] * 2.040816e-02f));
            }
        }
        write_channel_intel(avgpool_out_channel, tensor);
    }
}

__kernel void Conv2d_0c_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2, __global float *restrict compute)
{
    float input0[1024];
    for (int i = 0; i < 1024; i++){
        input0[i] = read_channel_intel(avgpool_out_channel);
    }
    for (int ff = 0; ff < 1001; ++ff)
    {
        compute[ff] = input2[ff];
        for (int rc = 0; rc < 1024; ++rc)
        {
            compute[ff] = (compute[ff] + (input0[rc] * input1[((ff * 1024) + rc)]));
        }
        compute[ff] = (compute[ff] > 0) ? compute[ff] : 0.0;
    }
}

// TODO InceptionV1/Logits/Conv2d_0c_1x1/Conv2D/Permute_

__kernel void Predictions_Reshape(__global float *restrict tensor, __global float *restrict input0, __global float *restrict input1)
{
    
    for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner)
    {
        tensor[ax0_ax1_fused_inner] = (input0[ax0_ax1_fused_inner] + input1[ax0_ax1_fused_inner]);
    }
}

__kernel void Predictions_Softmax(__global float *restrict input0,
                                  __global float *restrict tensor2)
{
    float tensor, tensor1;
    for (int ax1 = 0; ax1 < 1001; ++ax1)
    {
        tensor = -3.402823e+38f;
        for (int k1 = 0; k1 < 1001; ++k1)
        {
            tensor = max(tensor, input0[k1]);
        }
        tensor = 0.000000e+00f;
        for (int k2 = 0; k2 < 1001; ++k2)
        {
            tensor1 = (tensor1 + exp((input0[k2] - tensor)));
        }
        tensor2[ax1] = (exp((input0[ax1] - tensor)) / tensor1);
    }
}

__kernel void Predictions_Reshape_1(__global float *restrict T_reshape, __global float *restrict input0)
{
    for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner)
    {
        T_reshape[ax0_ax1_fused_inner] = input0[ax0_ax1_fused_inner];
    }
}
