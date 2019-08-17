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
channel concat_3b_struct concat_3c_in_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_3b_struct concat_3c_in_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_input_ch1"))); // Channel Rx
channel concat_3b_struct concat_3c_in_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_input_ch2"))); // Channel Rx
channel concat_3b_struct concat_3c_in_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_input_ch3"))); // Channel Rx

channel concat_3c_struct concat_3c_out_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel concat_3c_struct concat_3c_out_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_output_ch1"))); // Channel Tx
channel concat_3c_struct concat_3c_out_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_output_ch2"))); // Channel Tx
channel concat_3c_struct concat_3c_out_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_output_ch3"))); // Channel Tx

channel concat_3b_struct concat_3c_in_b0_channel __attribute__((depth(32))); // internal channel Branch 1
channel concat_3b_struct concat_3c_in_b1_channel __attribute__((depth(32))); // internal channel Branch 2
channel concat_3b_struct concat_3c_in_b2_channel __attribute__((depth(32))); // internal channel Branch 3
channel concat_3b_struct concat_3c_in_b3_channel __attribute__((depth(32))); // internal channel Branch 4

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
__kernel void feeder_3c(unsigned int route_from)
{

    for (int i = 0; i < 25088; i++)
    {
        struct concat_3b_buffer input;
        if (route_from == 0)
        {
            input = read_channel_intel(concat_3c_in_channel_0);
        }
        else if (route_from == 1)
        {
            input = read_channel_intel(concat_3c_in_channel_1);
        }
        else if (route_from == 2)
        {
            input = read_channel_intel(concat_3c_in_channel_2);
        }
        else // if (route_from == 3)
        {
            input = read_channel_intel(concat_3c_in_channel_3);
        }

        write_channel_intel(concat_3c_in_b0_channel, input);
        write_channel_intel(concat_3c_in_b1_channel, input);
        write_channel_intel(concat_3c_in_b2_channel, input);
        write_channel_intel(concat_3c_in_b3_channel, input);
    }
}

__kernel void Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{

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

        float local_input1[256];
        for (int k = 0; k < 256; k++){
            local_input1[k] = input1[((ff * 256) + k)];
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {

                float temp_0 = input2[ff];
                float temp_rc[256];
                #pragma unroll 4
                for (int rc = 0; rc < 256; rc++)
                {
                    temp_rc[rc] = (convInput[((((rc * 28) + yy) * 28) + xx)] * local_input1[rc]);
                }
                

                float temp_1 = 0.0;
                // #pragma unroll 2
                for (int rc = 0; rc < 256; rc++){
                    temp_1 += temp_rc[rc];
                }
                temp_0 += temp_1;

                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_3c_out_b0_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_3c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)

{

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
        
        float local_input1[256];
        for (int k = 0; k < 256; k++){
            local_input1[k] = input1[((ff * 256) + k)];
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                float temp_rc[256];
                #pragma unroll 4
                for (int rc = 0; rc < 256; rc++)
                {
                    temp_rc[rc] = (convInput[((((rc * 28) + yy) * 28) + xx)] * local_input1[rc]);
                }
                

                float temp_1 = 0.0;
                // #pragma unroll 2
                for (int rc = 0; rc < 256; rc++){
                    temp_1 += temp_rc[rc];
                }
                temp_0 += temp_1;

                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv2_1_3c_out_b1_channel, temp_0);
            }
        }
    }
}
__kernel void Padding_Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D()
{

    float input0[128 * 28 * 28];
    for (int i = 0; i < 128 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(conv2_1_3c_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_3c_out_b1_channel, (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f));
    }
}

__kernel void Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{
    float input0[115200];
    for (int i = 0; i < 115200; i++)
    {
        input0[i] = read_channel_intel(padding_3c_out_b1_channel);
    }

    for (int ff = 0; ff < 192; ++ff)
    {
        float local_input1[3*3*128];
        for (int k = 0; k < 3*3*128; k++){
            local_input1[k] = input1[((ff * 3*3*128) + k)];
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 128; ++rc)
                {
                    float temp_2 = 0.0;
                    // #pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_1 = 0.0;
                        #pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * local_input1[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                write_channel_intel(conv2_2_3c_out_b1_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_3c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{

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
        
        float local_input1[256];
        for (int k = 0; k < 256; k++){
            local_input1[k] = input1[((ff * 256) + k)];
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {

                float temp_0 = input2[ff];
                float temp_rc[256];
                #pragma unroll 4
                for (int rc = 0; rc < 256; rc++)
                {
                    temp_rc[rc] = (convInput[((((rc * 28) + yy) * 28) + xx)] * local_input1[rc]);
                }
                

                float temp_1 = 0.0;
                // #pragma unroll 2
                for (int rc = 0; rc < 256; rc++){
                    temp_1 += temp_rc[rc];
                }
                temp_0 += temp_1;

                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_3c_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Padding_Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[32 * 28 * 28];
    for (int i = 0; i < 32 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(conv3_1_3c_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28800; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_3c_out_b2_channel, (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f));
    }
}

__kernel void Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{

    float input0[28800];
    for (int i = 0; i < 28800; i++)
    {
        input0[i] = read_channel_intel(padding_3c_out_b2_channel);
    }

    for (int ff = 0; ff < 96; ++ff)
    {

        float local_input1[3*3*32];
        for (int k = 0; k < 3*3*32; k++){
            local_input1[k] = input1[((ff * 3*3*32) + k)];
        }

        #pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)
        {
            for (int xx = 0; xx < 28; ++xx)
            {
                float temp_0 = input2[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 32; ++rc)
                {

                    float temp_2 = 0.0;
                    // #pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {   
                        float temp_1 = 0.0;
                        #pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_1 += (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * local_input1[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.0;    
                write_channel_intel(conv3_2_3c_out_b2_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_3c_Branch_3_MaxPool_0a_3x3_MaxPool()
{

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

    #pragma loop_coalesce
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
}

__kernel void Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1,
                                                     __global float *restrict input2)
{

    float input0[256 * 28 * 28];
    for (int i = 0; i < 256 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(maxpool_3c_out_b3_channel);
    }

    for (int ff = 0; ff < 64; ++ff)
    {
        
        float local_input1[256];
        for (int k = 0; k < 256; k++){
            local_input1[k] = input1[((ff * 256) + k)];
        }

#pragma loop_coalesce
        for (int yy = 0; yy < 28; ++yy)

        {
            for (int xx = 0; xx < 28; ++xx)

            {
                float temp_0 = input2[ff];
                float temp_rc[256];
                #pragma unroll 2
                for (int rc = 0; rc < 256; rc++)
                {
                    temp_rc[rc] = (input0[((((rc * 28) + yy) * 28) + xx)] * local_input1[rc]);
                }
                

                float temp_1 = 0.0;
                // #pragma unroll 2
                for (int rc = 0; rc < 256; rc++){
                    temp_1 += temp_rc[rc];
                }
                temp_0 += temp_1;

                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv3_1_3c_out_b3_channel, temp_0);
            }
        }
    }
}

__kernel void Mixed_3c_concat(unsigned int route_to)
{
    //struct to store 256 bits of data
    struct concat_3c_buffer out;
    float input0[128 * 28 * 28];

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
        float result = (float)((326144 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -326144)] : (float)((250880 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -250880)] : (float)((100352 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -100352)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_3c_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            if (route_to == 0)
            {
                write_channel_intel(concat_3c_out_channel_0, out);
            }
            else if (route_to == 1)
            {
                write_channel_intel(concat_3c_out_channel_1, out);
            }
            else if (route_to == 2)
            {
                write_channel_intel(concat_3c_out_channel_2, out);
            }
            else if (route_to == 3)
            {
                write_channel_intel(concat_3c_out_channel_3, out);
            }
        }
    }
}
