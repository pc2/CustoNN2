/**
 * 4th Inception module - Inception 4c
 */
//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_4b_buffer {
    float concat_4b_out_buffer[8];
} concat_4b_struct;

typedef struct concat_4c_buffer {
    float concat_4c_out_buffer[8];
} concat_4c_struct;

// IO Channels for inception 4b to 4c
channel concat_4b_struct concat_4c_in_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_4b_struct concat_4c_in_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_input_ch1"))); // Channel Rx
channel concat_4b_struct concat_4c_in_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_input_ch2"))); // Channel Rx
channel concat_4b_struct concat_4c_in_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_input_ch3"))); // Channel Rx


channel concat_4c_struct concat_4c_out_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel concat_4c_struct concat_4c_out_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_output_ch1"))); // Channel Tx
channel concat_4c_struct concat_4c_out_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_output_ch2"))); // Channel Tx
channel concat_4c_struct concat_4c_out_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_output_ch3"))); // Channel Tx

channel concat_4b_struct concat_4c_in_b0_channel __attribute__((depth(32))) ; // internal channel Branch 1
channel concat_4b_struct concat_4c_in_b1_channel __attribute__((depth(32))) ; // internal channel Branch 2
channel concat_4b_struct concat_4c_in_b2_channel __attribute__((depth(32))) ; // internal channel Branch 3
channel concat_4b_struct concat_4c_in_b3_channel __attribute__((depth(32))) ; // internal channel Branch 4


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
__kernel void feeder_4c(unsigned int route_from)
{
    for (int i = 0; i < 12544; i++)
    {
        struct concat_4b_buffer input;
        if(route_from == 0)
        {
        input = read_channel_intel(concat_4c_in_channel_0);
        }
        else if (route_from == 1)
        {
        input = read_channel_intel(concat_4c_in_channel_1);
        }
        else if (route_from == 2)
        {
        input = read_channel_intel(concat_4c_in_channel_2);
        }
        else
        {
        input = read_channel_intel(concat_4c_in_channel_3);
        }                
        
        write_channel_intel(concat_4c_in_b0_channel, input);
        write_channel_intel(concat_4c_in_b1_channel, input);
        write_channel_intel(concat_4c_in_b2_channel, input);
        write_channel_intel(concat_4c_in_b3_channel, input);
    }
}


__kernel void Mixed_4c_Branch_0_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, constant float *restrict input2)
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
    //local memory for biases
    __local float input_bias[160];
    #pragma unroll 8
    for (int j = 0; j < 160; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 160; ++ff)
    {
        //local memory for weights
        float input_weight[512];
        #pragma unroll 8
        for (int k = 0; k < 512; k++){
            input_weight[k] = input1[((ff * 512) + k)];
        }
        
        float temp_out[14][14];
        for (int l = 0; l < 14; l++){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; ++rc)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = convInput[14*14*rc+i];
            }
#pragma unroll 4
        for (int yy = 0; yy < 14; ++yy)
        {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
            }
        }
    }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.0;
                write_channel_intel(conv1_4c_out_b0_channel, temp_out[yy][xx]);
            }
        }
    }
}


__kernel void Mixed_4c_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
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
    //local memory for biases
    __local float input_bias[112];
  //  #pragma unroll 32
    for (int j = 0; j < 112; j++){
        input_bias[j] = input2[j];
    }

    float l_input[196];
    for (int ff = 0; ff < 112; ++ff)
    {
        //local memory for weights
        float input_weight[512];
        #pragma unroll 8
        for (int k = 0; k < 512; k++){
            input_weight[k] = input1[((ff * 512) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; ++rc)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = convInput[14*14*rc+i];
            }
#pragma unroll 4
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
            }
                
            }
        }
        
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                write_channel_intel(conv2_1_4c_out_b1_channel, temp_out[yy][xx]);
            }
        }
    }
}

__kernel void Padding_Mixed_4c_Branch_1_Conv2d_0b_3x3_Conv2D()
{
    float input0[112*14*14];
    //#pragma unroll 64
    for (int i = 0; i < 112*14*14; i++){
        input0[i] = read_channel_intel(conv2_1_4c_out_b1_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28672; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4c_out_b1_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4c_Branch_1_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
    float input0[28672];
    for (int i = 0; i < 28672; i++){
        input0[i] = read_channel_intel(padding_4c_out_b1_channel);
    }
    //local memory for biases
    __local float input_bias[224];
   // #pragma unroll 32
    for (int j = 0; j < 224; j++){
        input_bias[j] = input2[j];
    }
    float l_input[16*16];
    for (int ff = 0; ff < 224; ++ff)
    {
        //local memory for weights
        float input_weight[3*3*112];
       #pragma unroll 16
        for (int k = 0; k < 3*3*112; k++){
            input_weight[k] = input1[((ff * 3*3*112) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
#pragma unroll 4
        for (int rc = 0; rc < 112; ++rc)
        {
            for (int i = 0; i < 16*16; i++){
                l_input[i] = input0[16*16*rc+i];
            }
#pragma unroll 2
        for (int yy = 0; yy < 14; ++yy)
        {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = 0;
#pragma unroll
                for (int rx = 0; rx < 3; ++rx)
                {
                    temp_0 += l_input[(yy+0) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 0) * 3) + rx)];
                }
                temp_out[yy][xx] += temp_0;
                
                float temp_1 = 0;
#pragma unroll
                for (int rx = 0; rx < 3; ++rx)
                {
                    temp_1 += l_input[(yy+1) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 1) * 3) + rx)];
                }
                temp_out[yy][xx] += temp_1;
                
                float temp_2 = 0;
#pragma unroll
                for (int rx = 0; rx < 3; ++rx)
                {
                    temp_2 += l_input[(yy+2) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 2) * 3) + rx)];
                }
                temp_out[yy][xx] += temp_2;
            }
        }
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.0;
                write_channel_intel(conv2_2_4c_out_b1_channel, temp_out[yy][xx]);
            }
        }
    }
}

__kernel void Mixed_4c_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
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
    //local memory for biases
    __local float input_bias[24];
   // #pragma unroll
    for (int j = 0; j < 24; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 24; ++ff)
    {
        //local memory for weights
        float input_weight[512];
       #pragma unroll 16
        for (int k = 0; k < 512; k++){
            input_weight[k] = input1[((ff * 512) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; ++rc)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = convInput[14*14*rc+i];
            }
#pragma unroll 4

        for (int yy = 0; yy < 14; ++yy)
        {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
            }
            
        }
        }
        
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                write_channel_intel(conv3_1_4c_out_b2_channel, temp_out[yy][xx]);
            }
        }
    }
}

__kernel void Padding_Mixed_4c_Branch_2_Conv2d_0b_3x3_Conv2D()
{
    float input0[24*14*14];
   // #pragma unroll 64
    for (int i = 0; i < 24*14*14; i++){
        input0[i] = read_channel_intel(conv3_1_4c_out_b2_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        write_channel_intel(padding_4c_out_b2_channel, (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] : 0.000000e+00f));
    }
}
__kernel void Mixed_4c_Branch_2_Conv2d_0b_3x3_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
    float input0[6144];
    for (int i = 0; i < 6144; i++){
        input0[i] = read_channel_intel(padding_4c_out_b2_channel);
    }
    //local memory for biases
    __local float input_bias[64];
   // #pragma unroll 32
    for (int j = 0; j < 64; j++){
        input_bias[j] = input2[j];
    }
    float l_input[16*16];
    for (int ff = 0; ff < 64; ++ff)
    {
        //local memory for weights
        float input_weight[3*3*24];
        #pragma unroll 16
        for (int k = 0; k < 3*3*24; k++){
            input_weight[k] = input1[((ff * 3*3*24) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
#pragma unroll 4
        for (int rc = 0; rc < 24; ++rc)
        {
            
            for (int i = 0; i < 16*16; i++){
                l_input[i] = input0[16*16*rc+i];
            }
            
#pragma unroll 2
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = 0;
                
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_0 += l_input[(yy+0) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 0) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_0;
                    
                    
                    float temp_1 = 0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_1 += l_input[(yy+1) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 1) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_1;
                    
                    float temp_2 = 0;
#pragma unroll
                    for (int rx = 0; rx < 3; ++rx)
                    {
                        temp_2 += l_input[(yy+2) * 16 + xx + rx] * input_weight[(((((rc) * 3) + 2) * 3) + rx)];
                    }
                    temp_out[yy][xx] += temp_2;
                }
            }
            }
            
            for (int yy = 0; yy < 14; ++yy)
            {
                for (int xx = 0; xx < 14; ++xx)
                {
                    temp_out[yy][xx] += input_bias[ff];
                    temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.0;
                    write_channel_intel(conv3_2_4c_out_b2_channel, temp_out[yy][xx]);
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
                #pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
                    #pragma unroll
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
                                                     constant float *restrict input2)
{
    float input0[512*14*14];
    for (int i = 0; i < 512*14*14; i++){
        input0[i] = read_channel_intel(maxpool_4c_out_b3_channel);
    }
    //local memory for biases
    __local float input_bias[64];
  //  #pragma unroll 8
    for (int j = 0; j < 64; j++){
        input_bias[j] = input2[j];
    }
    float l_input[196];
    for (int ff = 0; ff < 64; ++ff)
    {
        //local memory for weights
        float input_weight[512];
    #pragma unroll 8
        for (int k = 0; k < 512; k++){
            input_weight[k] = input1[((ff * 512) + k)];
        }
        float temp_out[14][14];
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 512; ++rc)
        {
            for (int i = 0; i < 14*14; i++){
                l_input[i] = input0[14*14*rc+i];
            }
            
#pragma unroll 4
            for (int yy = 0; yy < 14; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 14; ++xx)
                {
                    temp_out[yy][xx] += (l_input[yy * 14 + xx] * input_weight[rc]);
                }
                
            }
        }
        
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff];
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                write_channel_intel(conv4_1_4c_out_b3_channel, temp_out[yy][xx]);
            }
        }
    }
}


__kernel void Mixed_4c_concat(unsigned int route_to)
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
        //After accumlating 256 bits, send the data through IO channel.
        if (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 8 == 7)
        {
            if(route_to == 0)
            {
            write_channel_intel(concat_4c_out_channel_0, out);
            }
            else if (route_to == 1)
            {
            write_channel_intel(concat_4c_out_channel_1, out);
            }
            else if (route_to == 2)
            {
            write_channel_intel(concat_4c_out_channel_2, out);
            }
            else if (route_to == 3)
            {
            write_channel_intel(concat_4c_out_channel_3, out);
            }                        
            
        }

    }
}
