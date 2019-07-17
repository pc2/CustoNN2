/**
 * 9th Inception module - Inception 5c and Last layers
 */
//Enable the channel extension
 #pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_5b_buffer {
        float concat_5b_out_buffer[8];
} concat_5b_struct;


// IO Channels for inception 5b
channel concat_5b_struct concat_5b_in_channel __attribute__((depth(10))) __attribute__((io("kernel_input_ch0"))); // Channel Rx

channel concat_5b_struct concat_5b_in_b0_channel __attribute__((depth(10))) ; // internal channel Branch 1
channel concat_5b_struct concat_5b_in_b1_channel __attribute__((depth(10))) ; // internal channel Branch 2
channel concat_5b_struct concat_5b_in_b2_channel __attribute__((depth(10))) ; // internal channel Branch 3
channel concat_5b_struct concat_5b_in_b3_channel __attribute__((depth(10))) ; // internal channel Branch 4

//Auto run kernel to feed 5b results to 4 branches

__kernel void feeder_5b() {
	struct concat_5b_buffer input = read_channel_intel(concat_5b_in_channel);
	write_channel_intel(concat_5b_in_b0_channel, input);
    write_channel_intel(concat_5b_in_b1_channel, input);
    write_channel_intel(concat_5b_in_b2_channel, input);
    write_channel_intel(concat_5b_in_b3_channel, input);
}

__kernel void Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Read Input from IO channel 
    float convInput[40768];
    // 40768/8 = 5096
    for(int i=0;i<5096;i++)
    {
        //struct to store 256 bits of data
        struct concat_5b_buffer in;
        in = read_channel_intel(concat_5b_in_b0_channel);
        
        #pragma unroll
        for(int k=0;k<8;k++){
            convInput[(i*5096)+k] = in.concat_5b_out_buffer[k];
        }
    }
    for (int ff = 0; ff < 384; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (convInput[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
                }
                compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)]) > 0 ? (compute[((((ff * 7) + yy) * 7) + xx)]) : 0.000000e+00f;
            }
        }
    }
}

__kernel void Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Read Input from IO channel 
    float convInput[40768];
    // 40768/8 = 5096
    for(int i=0;i<5096;i++)
    {
        //struct to store 256 bits of data
        struct concat_5b_buffer in;
        in = read_channel_intel(concat_5b_in_b1_channel);
        
        #pragma unroll
        for(int k=0;k<8;k++){
            convInput[(i*5096)+k] = in.concat_5b_out_buffer[k];
        }
    }
    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (convInput[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
                }
                compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)]) > 0 ? compute[((((ff * 7) + yy) * 7) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Padding_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 15552; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
    }
}
__kernel void Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 384; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
                for (int rc = 0; rc < 192; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input1[((((((ff * 192) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] > 0) ? compute[((((ff * 7) + yy) * 7) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    //Read Input from IO channel 
    float convInput[40768];
    // 40768/8 = 5096
    for(int i=0;i<5096;i++)
    {
        //struct to store 256 bits of data
        struct concat_5b_buffer in;
        in = read_channel_intel(concat_5b_in_b2_channel);
        
        #pragma unroll
        for(int k=0;k<8;k++){
            convInput[(i*5096)+k] = in.concat_5b_out_buffer[k];
        }
    }
    for (int ff = 0; ff < 48; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (convInput[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
                }
                compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)]) > 0 ? compute[((((ff * 7) + yy) * 7) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Padding_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 3888; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] : 0.000000e+00f);
    }
}
__kernel void Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
                for (int rc = 0; rc < 48; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input1[((((((ff * 48) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] > 0) ? compute[((((ff * 7) + yy) * 7) + xx)] : 0.0;
            }
        }
    }
}

__kernel void Mixed_5c_Branch_3_MaxPool_0a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    //Read Input from IO channel 
    float maxInput[40768];
    // 40768/8 = 5096
    for(int i=0;i<5096;i++)
    {
        //struct to store 256 bits of data
        struct concat_5b_buffer in;
        in = read_channel_intel(concat_5b_in_b3_channel);
        
        #pragma unroll
        for(int k=0;k<8;k++){
            maxInput[(i*5096)+k] = in.concat_5b_out_buffer[k];
        }
    }
    for (int ax1 = 0; ax1 < 832; ++ax1)
    {
        for (int ax2 = 0; ax2 < 7; ++ax2)
        {
            for (int ax3 = 0; ax3 < 7; ++ax3)
            {
                tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (8 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (8 - rv1))) ? maxInput[(((((((ax1 * 7) + ax2) + rv) * 7) + ax3) + rv1) + -8)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 128; ++ff)
    {
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                compute[((((ff * 7) + yy) * 7) + xx)] = input2[ff];
                for (int rc = 0; rc < 832; ++rc)
                {
                    compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)]));
                }
                compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)]) > 0 ? compute[((((ff * 7) + yy) * 7) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Mixed_5c_concat(__global float *restrict T_transpose, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2, __global float *restrict input3)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((43904 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -43904)] : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] : (float)((18816 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -18816)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
    }
}

__kernel void AvgPool_0a_7x7_AvgPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 1024; ++ax1)
    {
        tensor[ax1] = 0.000000e+00f;
        for (int rv = 0; rv < 7; ++rv)
        {
            for (int rv1 = 0; rv1 < 7; ++rv1)
            {
                tensor[ax1] = (tensor[ax1] + (input0[((((ax1 * 7) + rv) * 7) + rv1)] * 2.040816e-02f));
            }
        }
    }
}

__kernel void Conv2d_0c_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
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