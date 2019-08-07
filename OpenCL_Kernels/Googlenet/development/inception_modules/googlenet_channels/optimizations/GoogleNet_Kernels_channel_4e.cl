/**
 * 6th Inception module - Inception 4e
 */
//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_4d_buffer
{
    float concat_4d_out_buffer[8];
} concat_4d_struct;

typedef struct concat_4e_buffer
{
    float concat_4e_out_buffer[8];
} concat_4e_struct;

// IO Channels for inception 4d to 4e
channel concat_4d_struct concat_4e_in_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_4d_struct concat_4e_in_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_input_ch1"))); // Channel Rx
channel concat_4d_struct concat_4e_in_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_input_ch2"))); // Channel Rx
channel concat_4d_struct concat_4e_in_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_input_ch3"))); // Channel Rx

channel concat_4e_struct concat_4e_out_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel concat_4e_struct concat_4e_out_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_output_ch1"))); // Channel Tx
channel concat_4e_struct concat_4e_out_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_output_ch2"))); // Channel Tx
channel concat_4e_struct concat_4e_out_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_output_ch3"))); // Channel Tx

channel concat_4d_struct concat_4e_in_b0_channel __attribute__((depth(32))); // internal channel Branch 1
channel concat_4d_struct concat_4e_in_b1_channel __attribute__((depth(32))); // internal channel Branch 2
channel concat_4d_struct concat_4e_in_b2_channel __attribute__((depth(32))); // internal channel Branch 3
channel concat_4d_struct concat_4e_in_b3_channel __attribute__((depth(32))); // internal channel Branch 4

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
    #pragma unroll 16
    for(int b = 0; b < 112; b++){
        input_bias[b] = input2[b];
    }
    for (int ff = 0; ff < 112; ++ff)
    {
		//Local weights 
        float input_weights[512];
        #pragma unroll 16
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
		
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0;
                #pragma unroll 4
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
    #pragma unroll 16
    for(int b = 0; b < 144; b++){
        input_bias[b] = input2[b];
    }
	
    for (int ff = 0; ff < 144; ++ff)
    {
	 //Local weights 
        float input_weights[512];
        #pragma unroll 16
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
			}
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                #pragma unroll 8 
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
    #pragma unroll 16
    for(int b = 0; b < 288; b++){
        input_bias[b] = input2[b];
    }
	
    for (int ff = 0; ff < 288; ++ff)
    {
	 //Local weights 
        float input_weights[3*3*144];
        #pragma unroll 16
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
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[((((((ff * 144) + rc) * 3) + ry) * 3) + rx)]);
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
    #pragma unroll 16
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
	}
    for (int ff = 0; ff < 32; ++ff)
    {
	//Local weights 
        float input_weights[512];
        #pragma unroll 16
        for(int m = 0 ; m < 512 ;m++){
            input_weights[m] = input1[((ff * 512) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                # pragma unroll 8
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
    #pragma unroll 16
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }
	
    for (int ff = 0; ff < 64; ++ff)
    {
		//Local weights 
        float input_weights[3*3*32];
        #pragma unroll 16
        for(int m = 0 ; m < 3*3*32 ; m++){
            input_weights[m] = input1[((ff * 3*3*32) + m)];
        }
        for (int yy = 0; yy < 14; ++yy)
        {
            for (int xx = 0; xx < 14; ++xx)
            {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
				# pragma unroll 32
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
                            temp_1 += (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input_weights[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]);
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
    #pragma unroll 16
    for(int b = 0; b < 64; b++){
        input_bias[b] = input2[b];
    }
	
    for (int ff = 0; ff < 64; ++ff)
    {
		//Local weights 
        float input_weights[512];
        #pragma unroll 16
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
