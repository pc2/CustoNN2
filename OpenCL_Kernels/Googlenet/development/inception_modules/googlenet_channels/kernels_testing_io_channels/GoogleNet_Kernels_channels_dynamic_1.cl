//First file
/**
 * 5th Inception module - Inception 4d
 */
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
channel concat_4c_struct concat_4d_in_channel_0 __attribute__((depth(10))) __attribute__((io("kernel_io_4c_ch0"))); // Channel Rx
channel concat_4c_struct concat_4d_in_channel_1 __attribute__((depth(10))) __attribute__((io("kernel_io_4c_ch1"))); // Channel Rx
channel concat_4c_struct concat_4d_in_channel_2 __attribute__((depth(10))) __attribute__((io("kernel_io_4c_ch2"))); // Channel Rx
channel concat_4c_struct concat_4d_in_channel_3 __attribute__((depth(10))) __attribute__((io("kernel_io_4c_ch3"))); // Channel Rx

channel concat_4d_struct concat_4d_out_channel_0 __attribute__((depth(10))) __attribute__((io("kernel_io_4d_ch0"))); // Channel Tx
channel concat_4d_struct concat_4d_out_channel_1 __attribute__((depth(10))) __attribute__((io("kernel_io_4d_ch1"))); // Channel Tx
channel concat_4d_struct concat_4d_out_channel_2 __attribute__((depth(10))) __attribute__((io("kernel_io_4d_ch2"))); // Channel Tx
channel concat_4d_struct concat_4d_out_channel_3 __attribute__((depth(10))) __attribute__((io("kernel_io_4d_ch3"))); // Channel Tx

channel concat_4c_struct concat_4d_in_b0_channel __attribute__((depth(10))); // internal channel Branch 1
channel concat_4c_struct concat_4d_in_b1_channel __attribute__((depth(10))); // internal channel Branch 2
channel concat_4c_struct concat_4d_in_b2_channel __attribute__((depth(10))); // internal channel Branch 3
channel concat_4c_struct concat_4d_in_b3_channel __attribute__((depth(10))); // internal channel Branch 4

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
__kernel void feeder_4d(unsigned int route_from)
{
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
}

__kernel void Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, constant float *restrict input2)
{
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
    __local  float input_bias[128];
    #pragma unroll 64
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 128; ++ff)
    {
        //Local weights 
        float input_weights[512];
        #pragma unroll 32
        for(int m = 0 ; m < 512 ;m++){
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
                write_channel_intel(conv1_4d_out_b0_channel, temp_0);
            }
        }
    }
}


__kernel void Mixed_4d_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, constant float *restrict input2)
{
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
    __local  float input_bias[128];
    #pragma unroll 64
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 128; ++ff)
    {
         //Local weights 
        float input_weights[512];
        #pragma unroll 32
        for(int m = 0 ; m < 512 ;m++){
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
}

__kernel void Padding_Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[128 * 14 * 14];
    for (int i = 0; i < 128 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv2_1_4d_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 32768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4d_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
    float input0[32768];
    for (int i = 0; i < 32768; i++)
    {
        input0[i] = read_channel_intel(padding_4d_out_b1_channel);
    }

    //Local memory for Biases:
    __local  float input_bias[256];
    #pragma unroll 32
    for(int b = 0; b < 256; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 256; ++ff)
    {
        //Local weights 
        float input_weights[3*3*128];
        #pragma unroll 32
        for(int m = 0 ; m < 3*3*128 ; m++){
            input_weights[m] = input1[((ff * 3*3*128) + m)];
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
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }

                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_4d_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4d_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
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
    __local  float input_bias[24];
    #pragma unroll 32
    for(int b = 0; b < 24; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 24; ++ff)
    {
        //Local weights 
        float input_weights[512];
        #pragma unroll 32
        for(int m = 0 ; m < 512 ;m++){
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
                    temp_1  += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_4d_out_b2_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[24 * 14 * 14];
    for (int i = 0; i < 24 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv3_1_4d_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4d_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
    float input0[6144];
    for (int i = 0; i < 6144; i++)
    {
        input0[i] = read_channel_intel(padding_4d_out_b2_channel);
    }

    //Local memory for Biases:
    __local  float input_bias[64];
    #pragma unroll 32
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 64; ++ff)
    {
         //Local weights 
        float input_weights[3*3*24];
        #pragma unroll 32
        for(int m = 0 ; m < 3*3*24 ; m++){
            input_weights[m] = input1[((ff * 3*3*24) + m)];
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
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 +=temp_3;
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
}

__kernel void Mixed_4d_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
    float input0[512 * 14 * 14];
    for (int i = 0; i < 512 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(maxpool_4d_out_b3_channel);
    }
    //Local memory for Biases:
    __local  float input_bias[64];
    #pragma unroll 64
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 64; ++ff)
    {
        //Local weights 
        float input_weights[512];
        #pragma unroll 32
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                 float temp_0 = input_bias[ff];
                float  temp_1 = 0.0;
                //#pragma unroll 8
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv4_1_4d_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4d_concat(unsigned int route_to)
{
    //struct to store 256 bits of data
    struct concat_4d_buffer out;

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
}


//Second file
/**
 * 6th Inception module - Inception 4e
 */
//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable



typedef struct concat_4e_buffer
{
    float concat_4e_out_buffer[8];
} concat_4e_struct;

// IO Channels for inception 4d to 4e
channel concat_4d_struct concat_4e_in_channel_0 __attribute__((depth(10))) __attribute__((io("kernel_io_4d_ch0"))); // Channel Rx
channel concat_4d_struct concat_4e_in_channel_1 __attribute__((depth(10))) __attribute__((io("kernel_io_4d_ch1"))); // Channel Rx
channel concat_4d_struct concat_4e_in_channel_2 __attribute__((depth(10))) __attribute__((io("kernel_io_4d_ch2"))); // Channel Rx
channel concat_4d_struct concat_4e_in_channel_3 __attribute__((depth(10))) __attribute__((io("kernel_io_4d_ch3"))); // Channel Rx

channel concat_4e_struct concat_4e_out_channel_0 __attribute__((depth(10))) __attribute__((io("kernel_io_4e_ch0"))); // Channel Tx
channel concat_4e_struct concat_4e_out_channel_1 __attribute__((depth(10))) __attribute__((io("kernel_io_4e_ch1"))); // Channel Tx
channel concat_4e_struct concat_4e_out_channel_2 __attribute__((depth(10))) __attribute__((io("kernel_io_4e_ch2"))); // Channel Tx
channel concat_4e_struct concat_4e_out_channel_3 __attribute__((depth(10))) __attribute__((io("kernel_io_4e_ch3"))); // Channel Tx

channel concat_4d_struct concat_4e_in_b0_channel __attribute__((depth(10))); // internal channel Branch 1
channel concat_4d_struct concat_4e_in_b1_channel __attribute__((depth(10))); // internal channel Branch 2
channel concat_4d_struct concat_4e_in_b2_channel __attribute__((depth(10))); // internal channel Branch 3
channel concat_4d_struct concat_4e_in_b3_channel __attribute__((depth(10))); // internal channel Branch 4

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
__kernel void feeder_4e(unsigned int route_from)
{
    for (int i = 0; i < 12544; i++)
    {
        struct concat_4d_buffer input;
        if (route_from == 0)
        {
            input = read_channel_intel(concat_4e_in_channel_0);
        }
        else if (route_from == 1)
        {
            input = read_channel_intel(concat_4e_in_channel_1);
        }
        else if (route_from == 2)
        {
            input = read_channel_intel(concat_4e_in_channel_2);
        }
        else //if (route_from == 3)
        {
            input = read_channel_intel(concat_4e_in_channel_3);
        }

        write_channel_intel(concat_4e_in_b0_channel, input);
        write_channel_intel(concat_4e_in_b1_channel, input);
        write_channel_intel(concat_4e_in_b2_channel, input);
        write_channel_intel(concat_4e_in_b3_channel, input);
    }
}

__kernel void Mixed_4e_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{

    //Read Input from IO channel
    float input0[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b0_channel);
		#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_4d_out_buffer[k];
        }
    }
    
	//Local memory for Biases:
    __local  float input_bias[112];
    #pragma unroll 128
    for(int b = 0; b < 112; b++){
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 112; ++ff)
    {
		//Local weights 
        float input_weights[512];
        #pragma unroll 64
        for(int m = 0 ; m < 512 ;m++){
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
				write_channel_intel(conv1_4e_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4e_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
    //Read Input from IO channel
    float input0[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b1_channel);
		#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_4d_out_buffer[k];
        }
    }
	
	//Local memory for Biases:
    __local  float input_bias[144];
    #pragma unroll 128
    for(int b = 0; b < 144; b++){
        input_bias[b] = input2[b];
    }
	
    for (int ff = 0; ff < 144; ++ff)
    {
	 //Local weights 
        float input_weights[512];
        #pragma unroll 64
        for(int m = 0 ; m < 512 ;m++){
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
                write_channel_intel(conv2_1_4e_out_b1_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[144 * 14 * 14];
    for (int i = 0; i < 144 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv2_1_4e_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 36864; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4e_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}

__kernel void Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
    float input0[36864];
    for (int i = 0; i < 36864; i++)
    {
        input0[i] = read_channel_intel(padding_4e_out_b1_channel);
    }
	
	//Local memory for Biases:
    __local  float input_bias[288];
    #pragma unroll 128
    for(int b = 0; b < 288; b++){
        input_bias[b] = input2[b];
    }
	
    for (int ff = 0; ff < 288; ++ff)
    {
	 //Local weights 
        float input_weights[3*3*144];
        #pragma unroll 128
        for(int m = 0 ; m < 3*3*144 ; m++){
            input_weights[m] = input1[((ff * 3*3*144) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;

                for (int rc = 0; rc < 144; ++rc)
                {
                    float temp_2 = 0.0;

		    #pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
			#pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }

                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
				temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_4e_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4e_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
    //Read Input from IO channel
    float input0[100352];
    // 100352/8 = 12544
    for (int i = 0; i < 12544; i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b2_channel);
		#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_4d_out_buffer[k];
        }
    }
	
	//Local memory for Biases:
    __local  float input_bias[32];
    #pragma unroll 32
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
	}
    for (int ff = 0; ff < 32; ++ff)
    {
	//Local weights 
        float input_weights[512];
        #pragma unroll 128
        for(int m = 0 ; m < 512 ;m++){
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
                write_channel_intel(conv3_1_4e_out_b2_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[32 * 14 * 14];
    for (int i = 0; i < 32 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv3_1_4e_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4e_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D(
    __global float *restrict input1,
    constant float *restrict input2)
{
    float input0[8192];
    for (int i = 0; i < 8192; i++)
    {
        input0[i] = read_channel_intel(padding_4e_out_b2_channel);
    }
	
	//Local memory for Biases:
    __local  float input_bias[64];
    #pragma unroll 
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }
	
    for (int ff = 0; ff < 64; ++ff)
    {
		//Local weights 
        float input_weights[3*3*32];
        #pragma unroll 128
        for(int m = 0 ; m < 3*3*32 ; m++){
            input_weights[m] = input1[((ff * 3*3*32) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 32; ++rc)
                {
					float temp_2 = 0.0;
                    #pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
						float temp_1 = 0.0;
                        #pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
						temp_2 += temp_1;
                    }
					temp_3 += temp_2;
                }
				temp_0 += temp_3;
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
    constant float *restrict input2)
{
    float input0[512 * 14 * 14];
    for (int i = 0; i < 512 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(maxpool_4e_out_b3_channel);
    }
	
	//Local memory for Biases:
    __local  float input_bias[64];
    #pragma unroll 64
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }
	
    for (int ff = 0; ff < 64; ++ff)
    {
		//Local weights 
        float input_weights[512];
        #pragma unroll 32
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float  temp_1 = 0.0;
                //#pragma unroll 8
                for (int rc = 0; rc < 512; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
				temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv4_1_4e_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4e_concat(unsigned int route_to)
{
    //struct to store 256 bits of data
    struct concat_4e_buffer out;

    float input0[112 * 14 * 14];
    for (int i = 0; i < 112 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv1_4e_out_b0_channel);
    }
    float input1[288 * 14 * 14];
    for (int i = 0; i < 288 * 14 * 14; i++)
    {
        input1[i] = read_channel_intel(conv2_2_4e_out_b1_channel);
    }
    float input2[64 * 14 * 14], input3[64 * 14 * 14];
    for (int i = 0; i < 64 * 14 * 14; i++)
    {
        input2[i] = read_channel_intel(conv3_2_4e_out_b2_channel);
        input3[i] = read_channel_intel(conv4_1_4e_out_b3_channel);
    }

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((90944 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -90944)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((21952 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -21952)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_4e_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            if (route_to == 0)
            {
                write_channel_intel(concat_4e_out_channel_0, out);
            }
            else if (route_to == 1)
            {
                write_channel_intel(concat_4e_out_channel_1, out);
            }
            else if (route_to == 2)
            {
                write_channel_intel(concat_4e_out_channel_2, out);
            }
            else if (route_to == 3)
            {
                write_channel_intel(concat_4e_out_channel_3, out);
            }
        }
    }
}

//Third file
/**
 * 7th Inception module - Inception 4f
 */
//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable



typedef struct concat_4f_buffer
{
    float concat_4f_out_buffer[8];
} concat_4f_struct;

// IO Channels for inception 4e to 4f
channel concat_4e_struct concat_4f_in_channel_0 __attribute__((depth(10))) __attribute__((io("kernel_io_4e_ch0"))); // Channel Rx
channel concat_4e_struct concat_4f_in_channel_1 __attribute__((depth(10))) __attribute__((io("kernel_io_4e_ch1"))); // Channel Rx
channel concat_4e_struct concat_4f_in_channel_2 __attribute__((depth(10))) __attribute__((io("kernel_io_4e_ch2"))); // Channel Rx
channel concat_4e_struct concat_4f_in_channel_3 __attribute__((depth(10))) __attribute__((io("kernel_io_4e_ch3"))); // Channel Rx

channel concat_4f_struct concat_4f_out_channel_0 __attribute__((depth(10))) __attribute__((io("kernel_io_4f_ch0"))); // Channel Tx
channel concat_4f_struct concat_4f_out_channel_1 __attribute__((depth(10))) __attribute__((io("kernel_io_4f_ch1"))); // Channel Tx
channel concat_4f_struct concat_4f_out_channel_2 __attribute__((depth(10))) __attribute__((io("kernel_io_4f_ch2"))); // Channel Tx
channel concat_4f_struct concat_4f_out_channel_3 __attribute__((depth(10))) __attribute__((io("kernel_io_4f_ch3"))); // Channel Tx

channel concat_4e_struct concat_4f_in_b0_channel __attribute__((depth(10))); // internal channel Branch 1
channel concat_4e_struct concat_4f_in_b1_channel __attribute__((depth(10))); // internal channel Branch 2
channel concat_4e_struct concat_4f_in_b2_channel __attribute__((depth(10))); // internal channel Branch 3
channel concat_4e_struct concat_4f_in_b3_channel __attribute__((depth(10))); // internal channel Branch 4

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
__kernel void feeder_4f(unsigned int route_from)
{
    for (int i = 0; i < 12936; i++)
    {
        struct concat_4e_buffer input;
        if (route_from == 0)
        {
            input = read_channel_intel(concat_4f_in_channel_0);
        }
        else if (route_from == 1)
        {
            input = read_channel_intel(concat_4f_in_channel_1);
        }
        else if (route_from == 2)
        {
            input = read_channel_intel(concat_4f_in_channel_2);
        }
        else // if (route_from == 3)
        {
            input = read_channel_intel(concat_4f_in_channel_3);
        }

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

    //Local memory for Biases:
    __local  float input_bias[256];
    #pragma unroll 64
    for(int b = 0; b < 256; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 256; ++ff)
    {
        //Local weights 
        float input_weights[528];
        #pragma unroll 32
        for(int m = 0 ; m < 528 ;m++){
            input_weights[m] = input1[((ff * 528) + m)];
        }


        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 528; ++rc)
                {
                    temp_1 += (convInput[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
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

    //Local memory for Biases:
    __local  float input_bias[160];
    #pragma unroll 64
    for(int b = 0; b < 160; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 160; ++ff)
    {
        //Local weights 
        float input_weights[528];
        #pragma unroll 32
        for(int m = 0 ; m < 528 ;m++){
            input_weights[m] = input1[((ff * 528) + m)];
        }

        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 528; ++rc)
                {
                    temp_1 += (convInput[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 +=temp_1;
                temp_0 = (temp_0 > 0) ? +temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_4f_out_b1_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_4f_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[160 * 14 * 14];
    for (int i = 0; i < 160 * 14 * 14; i++)
    {
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
    for (int i = 0; i < 40960; i++)
    {
        input0[i] = read_channel_intel(padding_4f_out_b1_channel);
    }
    //Local memory for Biases:
    __local  float input_bias[320];
    #pragma unroll 32
    for(int b = 0; b < 320; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 320; ++ff)
    {
        //Local weights 
        float input_weights[3*3*160];
        #pragma unroll 32
        for(int m = 0 ; m < 3*3*160 ; m++){
            input_weights[m] = input1[((ff * 3*3*160) + m)];
        }

        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 160; ++rc)
                {
                    float temp_2 = 0.0;
                    #pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {   
                        float temp_1 = 0.0;
                        #pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
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
    
    //Local memory for Biases:
    __local  float input_bias[32];
    #pragma unroll 32
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 32; ++ff)
    {
        //Local weights 
        float input_weights[528];
        #pragma unroll 32
        for(int m = 0 ; m < 528 ;m++){
            input_weights[m] = input1[((ff * 528) + m)];
        }

        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 528; ++rc)
                {
                    temp_1 += (convInput[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_4f_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Padding_Mixed_4f_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[32 * 14 * 14];
    for (int i = 0; i < 32 * 14 * 14; i++)
    {
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
    for (int i = 0; i < 8192; i++)
    {
        input0[i] = read_channel_intel(padding_4f_out_b2_channel);
    }
    //Local memory for Biases:
    __local  float input_bias[128];
    #pragma unroll 32
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }
    
    for (int ff = 0; ff < 128; ++ff)
    {
        //Local weights 
        float input_weights[3*3*32];
        #pragma unroll 32
        for(int m = 0 ; m < 3*3*32 ; m++){
            input_weights[m] = input1[((ff * 3*3*32) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 32; ++rc)
                {
                    float temp_2 = 0.0;
                    #pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {   
                        float temp_1 = 0.0;
                        #pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 +=temp_1;
                    }
                    temp_3 +=temp_2;
                }
                temp_0 += temp_3;
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
                #pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
                    #pragma unroll
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
    float input0[528 * 14 * 14];
    for (int i = 0; i < 528 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(maxpool_4f_out_b3_channel);
    }
    
    //Local memory for Biases:
    __local  float input_bias[128];
    #pragma unroll 64
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 128; ++ff)
    {
        //Local weights 
        float input_weights[528];
        #pragma unroll 32
        for(int m = 0 ; m < 528 ;m++){
            input_weights[m] = input1[((ff * 528) + m)];
        }

        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 528; ++rc)
                {
                    temp_1 += (input0[((((rc * 14) + yy) * 14) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv4_1_4f_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_4f_concat(unsigned int route_to)
{
    //struct to store 256 bits of data
    struct concat_4f_buffer out;

    float input0[256 * 14 * 14];
    for (int i = 0; i < 256 * 14 * 14; i++)
    {
        input0[i] = read_channel_intel(conv1_4f_out_b0_channel);
    }
    float input1[320 * 14 * 14];
    for (int i = 0; i < 320 * 14 * 14; i++)
    {
        input1[i] = read_channel_intel(conv2_2_4f_out_b1_channel);
    }
    float input2[128 * 14 * 14], input3[128 * 14 * 14];
    for (int i = 0; i < 128 * 14 * 14; i++)
    {
        input2[i] = read_channel_intel(conv3_2_4f_out_b2_channel);
        input3[i] = read_channel_intel(conv4_1_4f_out_b3_channel);
    }

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 163072; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((137984 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -137984)] : (float)((112896 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -112896)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_4f_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            if (route_to == 0)
            {
                write_channel_intel(concat_4f_out_channel_0, out);
            }
            else if (route_to == 1)
            {
                write_channel_intel(concat_4f_out_channel_1, out);
            }
            else if (route_to == 2)
            {
                write_channel_intel(concat_4f_out_channel_2, out);
            }
            else if (route_to == 3)
            {
                write_channel_intel(concat_4f_out_channel_3, out);
            }
        }
    }
}


//Fourth file
/**
 * 8th Inception module - 5a to mixed_5b_concat
 */
//enable channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable



typedef struct concat_5b_buffer
{
    float concat_5b_out_buffer[8];
} concat_5b_struct;

// IO Channels for inception 4f to 5a
channel concat_4f_struct concat_5a_in_channel_0 __attribute__((depth(10))) __attribute__((io("kernel_io_4f_ch0"))); // Channel Rx
channel concat_4f_struct concat_5a_in_channel_1 __attribute__((depth(10))) __attribute__((io("kernel_io_4f_ch1"))); // Channel Rx
channel concat_4f_struct concat_5a_in_channel_2 __attribute__((depth(10))) __attribute__((io("kernel_io_4f_ch2"))); // Channel Rx
channel concat_4f_struct concat_5a_in_channel_3 __attribute__((depth(10))) __attribute__((io("kernel_io_4f_ch3"))); // Channel Rx

channel concat_5b_struct concat_5b_out_channel_0 __attribute__((depth(10))) __attribute__((io("kernel_io_5b_ch0"))); // Channel Tx
channel concat_5b_struct concat_5b_out_channel_1 __attribute__((depth(10))) __attribute__((io("kernel_io_5b_ch1"))); // Channel Tx
channel concat_5b_struct concat_5b_out_channel_2 __attribute__((depth(10))) __attribute__((io("kernel_io_5b_ch2"))); // Channel Tx
channel concat_5b_struct concat_5b_out_channel_3 __attribute__((depth(10))) __attribute__((io("kernel_io_5b_ch3"))); // Channel Tx

channel concat_4f_struct concat_5a_in_max_channel __attribute__((depth(10))); // internal channel maxpool

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
    __local  float input_bias[256];
#pragma unroll 64
    for(int b = 0; b < 256; b++){
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 256; ++ff)
    {
        float input_weights[832];
#pragma unroll 128
        for(int w = 0 ; w < 832 ;w++){
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
    __local  float input_bias[160];
#pragma unroll 64
    for(int b = 0; b < 160; b++){
        input_bias[b] = input2[b];
    } 
    for (int ff = 0; ff < 160; ++ff)
    {
        float input_weights[832];
#pragma unroll 128
        for(int w = 0 ; w < 832 ;w++){
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
    __local  float input_bias[320];
#pragma unroll 64
    for(int b = 0; b < 320; b++){
        input_bias[b] = input2[b];
    } 
    for (int ff = 0; ff < 320; ++ff)
    {
        float input_weights[160*3*3];
#pragma unroll 128
        for(int w = 0 ; w < 160*3*3 ;w++){
            input_weights[w] = input1[((ff * 160*3*3) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3= 0.0;
                for (int rc = 0; rc < 160; ++rc)
                {
                    float temp_2 = 0.0;
                    #pragma unroll 
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
#pragma unroll 
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                            temp_2 += temp_1;
                        }
                        temp_3 += temp_2;
                    }
                temp_0 += temp_3;
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
    
    __local  float input_bias[32];
#pragma unroll 64
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
    } 
    for (int ff = 0; ff < 32; ++ff)
    {
        float input_weights[832];
#pragma unroll 128
        for(int w = 0 ; w < 832 ;w++){
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
    __local  float input_bias[128];
#pragma unroll 8
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    } 
    
    for (int ff = 0; ff < 128; ++ff)
    {
        float input_weights[32*3*3];
#pragma unroll 8
        for(int w = 0 ; w < 32*3*3 ;w++){
            input_weights[w] = input1[((ff * 32*3*3) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 32; ++rc)
                {
                float temp_2 = 0.0;
                    #pragma unroll 
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
                        #pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
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
    __local  float input_bias[128];
#pragma unroll 64
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 128; ++ff)
    {
        float input_weights[832];
#pragma unroll 128
        for(int w = 0 ; w < 832 ;w++){
            input_weights[w] = input1[((ff * 832) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 =0.0;
                
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_1 += (input0[((((rc * 7) + yy) * 7) + xx)] * input_weights[(rc)]);
                }
                
                temp_0 += temp_1;
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


/**
 * 9th Inception module - Inception 5c
 */
//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable



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


//Feeder kernels to read data from IO and feed it into internal channnels
__kernel void feeder_5c(unsigned int route_from)
{
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
}

__kernel void Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
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
    __local  float input_bias[384];
    #pragma unroll 128
    for(int b = 0; b < 384; b++){
        input_bias[b] = input2[b];
    }
     
    for (int ff = 0; ff < 384; ++ff)
    {
        float input_weights[832];
        #pragma unroll 128
        for(int w = 0 ; w < 832 ;w++){
            input_weights[w] = input1[((ff * 832) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1=0.0;
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_1 += (convInput[((((rc * 7) + yy) * 7) + xx)] * input_weights[(rc)]);
                }
                temp_0 += temp_1;
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
    __local  float input_bias[192];
    #pragma unroll 128
    for(int b = 0; b < 192; b++){
        input_bias[b] = input2[b];
    }
     
    for (int ff = 0; ff < 192; ++ff)
    {
        float input_weights[832];
        #pragma unroll 128
        for(int w = 0 ; w < 832 ;w++){
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
}

__kernel void Padding_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[192 * 7 * 7];
    for (int i = 0; i < 192 * 7 * 7; i++)
    {
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
    for (int i = 0; i < 15552; i++)
    {
        input0[i] = read_channel_intel(padding_5c_out_b1_channel);
    }
    __local  float input_bias[384];
    #pragma unroll 128
    for(int b = 0; b < 384; b++){
        input_bias[b] = input2[b];
    }
    
    for (int ff = 0; ff < 384; ++ff)
    {	
	float input_weights[192*3*3];
	#pragma unroll 128
        for(int w = 0 ; w < 192*3*3 ;w++){
            input_weights[w] = input1[((ff * 192*3*3) + w)];
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
                            temp_1 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
		            temp_3 += temp_2;
                }
	           	temp_0  += temp_3;
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
    __local  float input_bias[48];
    #pragma unroll 32
    for(int b = 0; b < 48; b++){
        input_bias[b] = input2[b];
    }
 
    for (int ff = 0; ff < 48; ++ff)
    {
        float input_weights[832];
        #pragma unroll 128
        for(int w = 0 ; w < 832 ;w++){
            input_weights[w] = input1[((ff * 832) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1=0.0;
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
}

__kernel void Padding_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[48 * 7 * 7];
    for (int i = 0; i < 48 * 7 * 7; i++)
    {
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
    for (int i = 0; i < 3888; i++)
    {
        input0[i] = read_channel_intel(padding_5c_out_b2_channel);
    }
    __local  float input_bias[128];
    #pragma unroll 32
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }
    
    for (int ff = 0; ff < 128; ++ff)
    {	float input_weights[48*3*3];
        #pragma unroll 32
        for(int w = 0 ; w < 48*3*3 ;w++){
            input_weights[w] = input1[((ff * 48*3*3) + w)];
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
                        float temp_1=0.0;
                        #pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input_weights[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        
                        temp_2 += temp_1;
                        
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
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
}

__kernel void Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[832 * 7 * 7];
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5c_out_b3_channel);
    }
    __local  float input_bias[128];
    #pragma unroll 32
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    } 
    for (int ff = 0; ff < 128; ++ff)
    {		
        float input_weights[832];
        #pragma unroll 32
        for(int w = 0 ; w < 832 ;w++){
            input_weights[w] = input1[((ff * 832) + w)];
        }
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1=0.0;
                for (int rc = 0; rc < 832; ++rc)
                {
                    temp_1 += (input0[((((rc * 7) + yy) * 7) + xx)] * input_weights[(rc)]);
                }
                
                temp_0 += temp_1;
                
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
    
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((43904 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -43904)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : (float)((18816 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -18816)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        write_channel_intel(concat_5c_out_channel, result);
    }
}

__kernel void AvgPool_0a_7x7_AvgPool()
{
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
}

__kernel void Conv2d_0c_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2, __global float *restrict compute)
{
    float input0[1024];
    for (int i = 0; i < 1024; i++)
    {
        input0[i] = read_channel_intel(avgpool_out_channel);
    }
    __local  float input_bias[1001];
    #pragma unroll 32
    for(int b = 0; b < 1001; b++){
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 1001; ++ff)
    {
	    float input_weights[1024];
    	#pragma unroll 32
    	for(int w = 0; w < 1024; w++){
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
}