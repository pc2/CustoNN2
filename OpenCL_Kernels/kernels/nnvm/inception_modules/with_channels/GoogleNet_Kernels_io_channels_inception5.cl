/**
 * 6th Inception module - Inception 4e
 */
//Enable the channel extension
 #pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_4d_buffer {
        float concat_4d_out_buffer[8];
} concat_4d_struct;

typedef struct concat_4e_buffer {
        float concat_4e_out_buffer[8];
} concat_4e_struct;

// IO Channels for inception 4d to 4e
channel concat_4d_struct concat_4e_in_channel __attribute__((depth(10))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_4e_struct concat_4e_out_channel __attribute__((depth(10))) __attribute__((io("kernel_output_ch0"))); // Channel Tx

channel concat_4d_struct concat_4e_in_b0_channel __attribute__((depth(10))) ; // internal channel Branch 1
channel concat_4d_struct concat_4e_in_b1_channel __attribute__((depth(10))) ; // internal channel Branch 2
channel concat_4d_struct concat_4e_in_b2_channel __attribute__((depth(10))) ; // internal channel Branch 3
channel concat_4d_struct concat_4e_in_b3_channel __attribute__((depth(10))) ; // internal channel Branch 4

//Auto run kernel to feed 3b results to 4 branches

__kernel void feeder_4e() {
	struct concat_4d_buffer input = read_channel_intel(concat_4e_in_channel);
	write_channel_intel(concat_4e_in_b0_channel, input);
    write_channel_intel(concat_4e_in_b1_channel, input);
    write_channel_intel(concat_4e_in_b2_channel, input);
    write_channel_intel(concat_4e_in_b3_channel, input);
}


__kernel void Mixed_4e_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel 
    float convInput[100352];
    // 100352/8 = 12544
    for(int i=0;i<12544;i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b0_channel);
        
        #pragma unroll
        for(int k=0;k<8;k++){
            convInput[(i*12544)+k] = in.concat_4d_out_buffer[k];
        }
    }
    for (int ff = 0; ff < 112; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Mixed_4e_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel 
    float convInput[100352];
    // 100352/8 = 12544
    for(int i=0;i<12544;i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b1_channel);
        
        #pragma unroll
        for(int k=0;k<8;k++){
            convInput[(i*12544)+k] = in.concat_4d_out_buffer[k];
        }
    }
    for (int ff = 0; ff < 144; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
            }
        }
    }
}
__kernel void Padding_Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 36864; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    for (int ff = 0; ff < 288; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 144; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 144) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_4e_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    //Read Input from IO channel 
    float convInput[100352];
    // 100352/8 = 12544
    for(int i=0;i<12544;i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b2_channel);
        
        #pragma unroll
        for(int k=0;k<8;k++){
            convInput[(i*12544)+k] = in.concat_4d_out_buffer[k];
        }
    }
    for (int ff = 0; ff < 32; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (convInput[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
            }
        }
    }
}
__kernel void Padding_Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f);
    }
}
__kernel void Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 32; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_4e_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    //Read Input from IO channel 
    float maxInput[100352];
    // 100352/8 = 12544
    for(int i=0;i<12544;i++)
    {
        //struct to store 256 bits of data
        struct concat_4d_buffer in;
        in = read_channel_intel(concat_4e_in_b3_channel);
        
        #pragma unroll
        for(int k=0;k<8;k++){
            maxInput[(i*12544)+k] = in.concat_4d_out_buffer[k];
        }
    }
    for (int ax1 = 0; ax1 < 512; ++ax1)
    {
        for (int ax2 = 0; ax2 < 14; ++ax2)
        {
            for (int ax3 = 0; ax3 < 14; ++ax3)
            {
                tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? maxInput[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_4e_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute,
                                                     __global float *restrict input0,
                                                     __global float *restrict input1,
                                                     __global float *restrict input2)
{
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
                for (int rc = 0; rc < 512; ++rc)
                {
                    compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
                }
                compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Mixed_4e_concat(__global float *restrict T_concat, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    //struct to store 256 bits of data
    struct concat_4e_buffer out;
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        //T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((90944 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -90944)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((21952 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -21952)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        float result = (float)((90944 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -90944)] : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] : (float)((21952 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -21952)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_4e_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner%8] = result;
        //After accumlating 256 bits, send the data through IO channel.
        if(ax0_ax1_fused_ax2_fused_ax3_fused_inner%8 == 7){
            write_channel_intel(concat_4e_out_channel,out);
        }
    }
}