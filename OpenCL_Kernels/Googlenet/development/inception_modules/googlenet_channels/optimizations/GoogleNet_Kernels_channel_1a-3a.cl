//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable
//256 bits io channel struct

typedef struct concat_3a_buffer
{
    float concat_3a_out_buffer[8];
} concat_3a_struct;
// IO Channels for inception 3b to 3c
channel concat_3a_struct concat_3a_out_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel concat_3a_struct concat_3a_out_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_output_ch1"))); // Channel Tx
channel concat_3a_struct concat_3a_out_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_output_ch2"))); // Channel Tx
channel concat_3a_struct concat_3a_out_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_output_ch3"))); // Channel Tx

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
    for (int i = 0; i < 157323; i++)
    {
        input0[i] = read_channel_intel(padding_1a_out_channel);
    }
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 112; ++yy)
        {
            for (int xx = 0; xx < 112; ++xx)
            {
                float temp_0 = input2[ff];
                float temp_3 = 0;
                for (int rc = 0; rc < 3; ++rc)
                {
                    float temp_2 = 0;
                    //#pragma unroll
                    for (int ry = 0; ry < 7; ++ry)
                    {
                        float temp_1 = 0;
                        #pragma unroll
                        for (int rx = 0; rx < 7; ++rx)
                        {
                            temp_1 += (input0[(((((rc * 52441) + (yy * 458)) + (ry * 229)) + (xx * 2)) + rx)] * input1[((((((ff * 3) + rc) * 7) + ry) * 7) + rx)]);
                        }
                        temp_2 +=temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_1a_out_channel, temp_0);
            }
        }
    }
}

__kernel void MaxPool_2a_3x3_MaxPool()
{

    

    for (int ax1 = 0; ax1 < 64; ++ax1)
    {
        //Store 1 slice of data
        float input0[112 * 112];
        for (int i = 0; i < 112 * 112; i++)
        {
            input0[i] = read_channel_intel(conv1_1a_out_channel);
        }
        for (int ax2 = 0; ax2 < 56; ++ax2)
        {
            for (int ax3 = 0; ax3 < 56; ++ax3)
            {
                float tensor = -3.402823e+38f;
                #pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
                    #pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((ax2 * 2) < (112 - rv)) && ((ax3 * 2) < (112 - rv1))) ? input0[(((((((ax2) * 2) + rv) * 56) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
                write_channel_intel(maxpool_2a_out_channel, tensor);
            }
        }
    }
}

# define SR 36
__kernel void Conv2d_2b_1x1_Conv2D(__global float *restrict input1, __global float *restrict input2)
{

    float input0[64 * 56 * 56];
    for (int i = 0; i < 64 * 56 * 56; i++)
    {
        input0[i] = read_channel_intel(maxpool_2a_out_channel);
    }
    
   

    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 56; ++yy)
        {
            for (int xx = 0; xx < 56; ++xx)
            {
                 float temp_copies[SR];

	            //Initialize the array with 0
	            #pragma unroll
	            for(int i=0;i<SR;i++)
		            temp_copies[i]=0.0;
                
                float temp_0 = input2[ff];
                float temp_1 = 0;
                #pragma unroll 8
                for (int rc = 0; rc < 64; ++rc)
                {
                    //temp_1 += (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 64) + rc)]);
                    float temp = temp_copies[SR-1] + (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 64) + rc)]);

                    #pragma unroll
                    for (unsigned j = SR-1; j >0; j--)
                        temp_copies[j] = temp_copies[j-1];
                    
                    temp_copies[0] = temp;

                }
                #pragma unroll
                for (unsigned i= 0; i< SR; i++)
                    temp_1 += temp_copies[i];
                
                
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_2b_out_channel, temp_0);
            }
        }
    }
}

__kernel void Padding_Conv2d_2c_3x3_Conv2D()
{

    float input0[64 * 56 * 56];
    for (int i = 0; i < 64 * 56 * 56; i++)
    {
        input0[i] = read_channel_intel(conv1_2b_out_channel);
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        float result = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f);
        write_channel_intel(padding_2c_out_channel, result);
    }
}

__kernel void Conv2d_2c_3x3_Conv2D(__global float *restrict input1, __global float *restrict input2)
{

    float input0[215296];
    for (int i = 0; i < 215296; i++)
    {
        input0[i] = read_channel_intel(padding_2c_out_channel);
    }
    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 56; ++yy)
        {
            for (int xx = 0; xx < 56; ++xx)
            {
                float temp_0 = input2[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 64; ++rc)
                {
                    float temp_2 = 0;
                    //#pragma unroll
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        float temp_3 = 0;
                        #pragma unroll
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            temp_3 += (input0[((((((rc * 58) + yy) + ry) * 58) + xx) + rx)] * input1[((((((ff * 64) + rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 +=temp_3;
                    }
                    temp_1 += temp_2;
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                write_channel_intel(conv1_2c_out_channel, temp_0);
            }
        }
    }
}

//last kernel to be launched in FPGA needs to route data out of FPGA according to routing logic
//determined in plugin
__kernel void MaxPool_3a_3x3_MaxPool(unsigned int route_to)
{

    
    //struct to store 256 bits of data
    struct concat_3a_buffer out;

    for (int ax1 = 0; ax1 < 192; ++ax1)
    {
        float input0[56 * 56];
        for (int i = 0; i < 56 * 56; i++)
        {
            input0[i] = read_channel_intel(conv1_2c_out_channel);
        }
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {
                float tensor = -3.402823e+38f;
                #pragma unroll
                for (int rv = 0; rv < 3; ++rv)
                {
                    #pragma unroll
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor = max(tensor, (float)((((ax2 * 2) < (56 - rv)) && ((ax3 * 2) < (56 - rv1))) ? input0[((((((( ax2) * 2) + rv) * 28) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
                out.concat_3a_out_buffer[((((ax1 * 28) + ax2) * 28) + ax3) % 8] = tensor;
                //After accumlating 256 bits, send the data through IO channel.
                if (((((ax1 * 28) + ax2) * 28) + ax3) % 8 == 7)
                {
                    //route to different IO channels depending on topology determined in plugin
                    if (route_to == 0)
                    {
                        write_channel_intel(concat_3a_out_channel_0, out);
                    }
                    else if (route_to == 1)
                    {
                        write_channel_intel(concat_3a_out_channel_1, out);
                    }
                    else if (route_to == 2)
                    {
                        write_channel_intel(concat_3a_out_channel_2, out);
                    }
                    else if (route_to == 3)
                    {
                        write_channel_intel(concat_3a_out_channel_3, out);
                    }                    
                }
            }
        }
    }
}
