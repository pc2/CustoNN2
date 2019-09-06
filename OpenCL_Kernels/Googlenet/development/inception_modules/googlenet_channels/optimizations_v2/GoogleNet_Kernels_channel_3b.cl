//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_3a_buffer
{
    float concat_3a_out_buffer[8];
} concat_3a_struct;

typedef struct concat_3b_buffer
{
    float concat_3b_out_buffer[8];
} concat_3b_struct;

// IO Channels for inception 3b to 3c
channel concat_3a_struct concat_3a_in_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_3a_struct concat_3a_in_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_input_ch1"))); // Channel Rx
channel concat_3a_struct concat_3a_in_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_input_ch2"))); // Channel Rx
channel concat_3a_struct concat_3a_in_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_input_ch3"))); // Channel Rx

channel concat_3b_struct concat_3b_out_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel concat_3b_struct concat_3b_out_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_output_ch1"))); // Channel Tx
channel concat_3b_struct concat_3b_out_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_output_ch2"))); // Channel Tx
channel concat_3b_struct concat_3b_out_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_output_ch3"))); // Channel Tx

channel concat_3a_struct concat_3b_in_b0_channel __attribute__((depth(32))); // internal channel Branch 1
channel concat_3a_struct concat_3b_in_b1_channel __attribute__((depth(32))); // internal channel Branch 2
channel concat_3a_struct concat_3b_in_b2_channel __attribute__((depth(32))); // internal channel Branch 3
channel concat_3a_struct concat_3b_in_b3_channel __attribute__((depth(32))); // internal channel Branch 4

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
//Added an argument to decide from which channel to read
__kernel void feeder_3b(unsigned int route_from)
{
    for (int i = 0; i < 18816; i++)
    {
        struct concat_3a_buffer input;
        if (route_from == 0)
        {
            input = read_channel_intel(concat_3a_in_channel_0);
        }
        else if (route_from == 1)
        {
            input = read_channel_intel(concat_3a_in_channel_1);
        }
        else if (route_from == 2)
        {
            input = read_channel_intel(concat_3a_in_channel_2);
        }
        else // if (route_from == 3)
        {
            input = read_channel_intel(concat_3a_in_channel_3);
        }

        write_channel_intel(concat_3b_in_b0_channel, input);
        write_channel_intel(concat_3b_in_b1_channel, input);
        write_channel_intel(concat_3b_in_b2_channel, input);
        write_channel_intel(concat_3b_in_b3_channel, input);
    }
}

__kernel void Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{

    float input0[192 * 28 * 28];
    for (int i = 0; i < 18816; i++)
    {
        struct concat_3a_buffer in;
        in = read_channel_intel(concat_3b_in_b0_channel);
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_3a_out_buffer[k];
        }
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
        float input_weights[192];
        #pragma unroll 128
        for(int m = 0 ; m < 192 ;m++){
            input_weights[m] = input1[((ff * 192) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 192; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[28*28];
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }
            #pragma unroll 2
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += (image_slice[(yy * 28) + xx] * input_weights[rc]);
                }

            }
        }
            //Summarize the results depthwise.
            #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += input_bias[ff];
                    //Relu
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    write_channel_intel(conv1_3b_out_b0_channel, temp_out[yy][xx]);
                }
            }
    }
}

__kernel void Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{

    float input0[192 * 28 * 28];
    for (int i = 0; i < 18816; i++)
    {
        struct concat_3a_buffer in;
        in = read_channel_intel(concat_3b_in_b1_channel);
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_3a_out_buffer[k];
        }
    }

     //Local memory for Biases:
    __local  float input_bias[96];
    #pragma unroll 32
    for(int b = 0; b < 96; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 96; ++ff)
    {
        //Local weights 
        float input_weights[192];
        #pragma unroll 64
        for(int m = 0 ; m < 192 ;m++){
            input_weights[m] = input1[((ff * 192) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 192; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[28*28];
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }
            #pragma unroll 2
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += (image_slice[(yy * 28) + xx] * input_weights[rc]);
                }

            }
        }
        //Summarize the results depthwise.
        #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += input_bias[ff];
                    //Relu
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    write_channel_intel(conv2_1_3b_out_b1_channel, temp_out[yy][xx]);
                }
            }

    }
}

__kernel void Padding_Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D()
{

    float input0[96 * 28 * 28];
    for (int i = 0; i < 96 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(conv2_1_3b_out_b1_channel);
    }

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 86400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
        write_channel_intel(padding_3b_out_b1_channel, result);
    }
}

__kernel void Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{

    float input0[86400];
    for (int i = 0; i < 86400; i++)
    {
        input0[i] = read_channel_intel(padding_3b_out_b1_channel);
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
        float local_weight[3*3*96];
        #pragma unroll 32
        for(int m = 0 ; m < 3*3*96 ; m++){
            local_weight[m] = input1[((ff * 3*3*96) + m)];
        }

        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0.0;
            }
        }

        for (int rc = 0; rc < 96; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[30*30];
            for (int in = 0; in < 30*30; in++){
                image_slice[in] = input0[(30*30*rc)+in];
            }
             //Convultion 3*3
            #pragma unroll 2
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll 
                for (int xx = 0; xx < 28; ++xx)
                {
                        float temp_0 = 0;
                       
                        float temp_2 = 0;
                        #pragma unroll
                        for (int ry = 0; ry < 3; ++ry)
                        {
                            float temp_1 = 0;
                            #pragma unroll
                            for (int rx = 0; rx < 3; ++rx)
                            {
                                temp_1 +=  (image_slice[((yy+ry) * 30) + (xx) + rx ] * local_weight[(((((rc) * 3) + ry) * 3) + rx)]);
                            }
                            temp_2 +=temp_1;
                        }
                        temp_0 += temp_2;
                        temp_out[yy][xx] += temp_0;
                }
            }
        }
            //Summarize the results depthwise.
             #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += input_bias[ff];
                    //RELU
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f; 
                    write_channel_intel(conv2_2_3b_out_b1_channel, temp_out[yy][xx]);
                }
            }

    }
}

__kernel void Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{

    float input0[192 * 28 * 28];
    for (int i = 0; i < 18816; i++)
    {
        struct concat_3a_buffer in;
        in = read_channel_intel(concat_3b_in_b2_channel);
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_3a_out_buffer[k];
        }
    }

    
     //Local memory for Biases:
    __local  float input_bias[16];
    #pragma unroll
    for(int b = 0; b < 16; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 16; ++ff)
    {
        //Local weights 
        float input_weights[192];
        #pragma unroll 64
        for(int m = 0 ; m < 192 ;m++){
            input_weights[m] = input1[((ff * 192) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 192; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[28*28];
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }
            //#pragma unroll 4
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll 8
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += (image_slice[yy * 28 + xx] * input_weights[rc]);
                }
            }
        }
            //Summarize the results depthwise.
            #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += input_bias[ff];
                    //Relu
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    write_channel_intel(conv3_1_3b_out_b2_channel, temp_out[yy][xx]);
                }
            }

        
    }
}

__kernel void Padding_Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D()
{

    float input0[16 * 28 * 28];
    for (int i = 0; i < 16 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(conv3_1_3b_out_b2_channel);
    }

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 14400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
        write_channel_intel(padding_3b_out_b2_channel, result);
    }
}

__kernel void Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{

    float input0[14400];
    for (int i = 0; i < 14400; i++)
    {
        input0[i] = read_channel_intel(padding_3b_out_b2_channel);
    }

     //Local memory for Biases:
    __local  float input_bias[32];
    #pragma unroll
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
    }

    for (int ff = 0; ff < 32; ++ff)
    {
        //Local weights 
        float local_weight[3*3*16];
        #pragma unroll 64
        for(int m = 0 ; m < 3*3*16 ; m++){
            local_weight[m] = input1[((ff * 3*3*16) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0.0;
            }
        }
        for (int rc = 0; rc < 16; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[30*30];
            for (int in = 0; in < 30*30; in++){
                image_slice[in] = input0[(30*30*rc)+in];
            }
            //#pragma unroll 4
            for (int yy = 0; yy < 28; ++yy)
            {
                #pragma unroll 8
                for (int xx = 0; xx < 28; ++xx)
                {
                    float temp_0 = 0;
                        //Convultion 3*3
                        float temp_2 = 0;
                        #pragma unroll
                        for (int ry = 0; ry < 3; ++ry)
                        {
                            float temp_1 = 0;
                            #pragma unroll
                            for (int rx = 0; rx < 3; ++rx)
                            {
                                temp_1 +=  (image_slice[((yy+ry) * 30) + (xx) + rx ] * local_weight[(((((rc) * 3) + ry) * 3) + rx)]);
                            }
                            temp_2 +=temp_1;
                        }
                        temp_0 += temp_2;
                        temp_out[yy][xx] += temp_0;
                }
            }
        }
            //Summarize the results depthwise.
             #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += input_bias[ff];
                    //RELU
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f; 
                    write_channel_intel(conv3_2_3b_out_b2_channel, temp_out[yy][xx]);
                }
            }
    }
}

__kernel void Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool()
{

    float input0[192 * 28 * 28];
    for (int i = 0; i < 18816; i++)
    {
        struct concat_3a_buffer in;
        in = read_channel_intel(concat_3b_in_b3_channel);
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            input0[(i * 8) + k] = in.concat_3a_out_buffer[k];
        }
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

    float input0[192 * 28 * 28];
    for (int i = 0; i < 192 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(maxpool_3b_out_b3_channel);
    }

    //Local memory for Biases:
    __local  float input_bias[32];
    #pragma unroll
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
    }


    for (int ff = 0; ff < 32; ++ff)
    {
        //Local weights 
        float input_weights[192];
        #pragma unroll 64
        for(int m = 0 ; m < 192 ;m++){
            input_weights[m] = input1[((ff * 192) + m)];
        }
        //2D array to store Temporary results of 1 slice.
        float temp_out[28][28];
        //Initialize values with 0
        #pragma loop_coalesce
        for (int l = 0; l < 28; l++ ){
            for (int j = 0; j < 28; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 192; ++rc)
        {
            //Store 1 slice of input image
            float image_slice[28*28];
            for (int in = 0; in < 28*28; in++){
                image_slice[in] = input0[(28*28*rc)+in];
            }
            //#pragma unroll 4
            for (int yy = 0; yy < 28; ++yy)
            {
                //#pragma unroll
                for (int xx = 0; xx < 28; ++xx)
                {
                    temp_out[yy][xx] += (image_slice[yy * 28 + xx] * input_weights[rc]);
                }
            }
        }
            //Summarize the results depthwise.
            #pragma loop_coalesce
            for (int yy = 0; yy < 28; ++yy)
            {
                for (int xx = 0; xx < 28; ++xx)
                {   
                    temp_out[yy][xx] += input_bias[ff];
                    //Relu
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                    write_channel_intel(conv3_1_3b_out_b3_channel, temp_out[yy][xx]);
                }
            }
    }
}

__kernel void Mixed_3b_concat(unsigned int route_to)
{

    //struct to store 256 bits of data
    struct concat_3b_buffer out;

    float input0[64 * 28 * 28];
    for (int i = 0; i < 64 * 28 * 28; i++)
    {
        input0[i] = read_channel_intel(conv1_3b_out_b0_channel);
    }
    float input1[128 * 28 * 28];
    for (int i = 0; i < 128 * 28 * 28; i++)
    {
        input1[i] = read_channel_intel(conv2_2_3b_out_b1_channel);
    }
    float input2[32 * 28 * 28], input3[32 * 28 * 28];
    for (int i = 0; i < 32 * 28 * 28; i++)
    {
        input2[i] = read_channel_intel(conv3_2_3b_out_b2_channel);
        input3[i] = read_channel_intel(conv3_1_3b_out_b3_channel);
    }

    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)((175616 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -175616)] : (float)((150528 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -150528)] : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] : input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner])));
        out.concat_3b_out_buffer[ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8] = result;
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            //route to different channels depending on topology determined at plug in
            if (route_to == 0)
            {
                write_channel_intel(concat_3b_out_channel_0, out);
            }
            else if (route_to == 1)
            {
                write_channel_intel(concat_3b_out_channel_1, out);
            }
            else if (route_to == 2)
            {
                write_channel_intel(concat_3b_out_channel_2, out);
            }
            else if (route_to == 3)
            {
                write_channel_intel(concat_3b_out_channel_3, out);
            }
        }
    }
}
