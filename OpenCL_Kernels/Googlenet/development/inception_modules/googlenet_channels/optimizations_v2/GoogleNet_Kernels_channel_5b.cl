/**
 * 8th Inception module - 5a to mixed_5b_concat
 */
//enable channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//256 bits io channel struct
typedef struct concat_4f_buffer
    {
        float concat_4f_out_buffer[8];
    } concat_4f_struct;

typedef struct concat_5b_buffer
    {
        float concat_5b_out_buffer[8];
    } concat_5b_struct;

// IO Channels for inception 4f to 5a
channel concat_4f_struct concat_5a_in_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_input_ch0"))); // Channel Rx
channel concat_4f_struct concat_5a_in_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_input_ch1"))); // Channel Rx
channel concat_4f_struct concat_5a_in_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_input_ch2"))); // Channel Rx
channel concat_4f_struct concat_5a_in_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_input_ch3"))); // Channel Rx

channel concat_5b_struct concat_5b_out_channel_0 __attribute__((depth(8))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel concat_5b_struct concat_5b_out_channel_1 __attribute__((depth(8))) __attribute__((io("kernel_output_ch1"))); // Channel Tx
channel concat_5b_struct concat_5b_out_channel_2 __attribute__((depth(8))) __attribute__((io("kernel_output_ch2"))); // Channel Tx
channel concat_5b_struct concat_5b_out_channel_3 __attribute__((depth(8))) __attribute__((io("kernel_output_ch3"))); // Channel Tx

channel concat_4f_struct concat_5a_in_max_channel __attribute__((depth(32))); // internal channel maxpool

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
                                                     constant float *restrict input2)
{
    float input0[832 * 7 * 7];
    
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5a_out_channel1);
    }
    //Local memory for Biases:
    __local  float input_bias[256];
    
    for(int b = 0; b < 256; b++){
        input_bias[b] = input2[b];
    }

    float l_input[49];
       for (int ff = 0; ff < 256; ++ff)
    {
        //Local weights 
        float input_weights[832];
		#pragma unroll 8
        for(int m = 0 ; m < 832 ;m++){
            input_weights[m] = input1[((ff * 832) + m)];
        }
        
        float temp_out[7][7];
        #pragma loop_coalesce
        for (int l = 0; l < 7; l++ ){
            for (int j = 0; j < 7; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 832; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            
#pragma unroll 4
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[ yy * 7 + xx] * input_weights[rc]);
                }
             
                
            }
        }
        #pragma loop_coalesce
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff]; 
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                write_channel_intel(conv1_5b_out_b0_channel, temp_out[yy][xx]);
            }
        }
    }
}

__kernel void Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float *restrict input1,
                                                     constant float *restrict input2)
{
    float input0[832 * 7 * 7];

    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5a_out_channel2);
    }
    //Local memory for Biases:
    __local  float input_bias[160];
    
    for(int b = 0; b < 160; b++){
        input_bias[b] = input2[b];
    }

    float l_input[49];
       for (int ff = 0; ff < 160; ++ff)
    {
        //Local weights 
        float input_weights[832];
		#pragma unroll 8
        for(int m = 0 ; m < 832 ;m++){
            input_weights[m] = input1[((ff * 832) + m)];
        }
        
        float temp_out[7][7];
        #pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 832; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            
#pragma unroll 4
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[ yy * 14 + xx] * input_weights[rc]);
                }
             
                
            }
        }
        #pragma loop_coalesce
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff]; 
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                write_channel_intel(conv2_1_5b_out_b1_channel, temp_out[yy][xx]);
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
                                                   constant float *restrict input2)
{
  float input0[12960];
 
  for (int i = 0; i < 12960; i++)
  {
      input0[i] = read_channel_intel(padding_5b_out_b1_channel);
  }
  __local  float input_bias[320];

  for(int b = 0; b < 320; b++){
      input_bias[b] = input2[b];
  }
  float l_input[9*9];
  for (int ff = 0; ff < 320; ++ff)
  {
      float input_weights[3*3*160];
      #pragma unroll 8
   for (int k = 0; k < 3*3*160; k++){
          input_weights[k] = input1[((ff *3*3*160) + k)];
   }
   float temp_out[7][7];
      #pragma loop_coalesce
      for (int l = 0; l < 7; l++ ){
          for (int j = 0; j < 7; j++){
              temp_out[l][j] = 0.0;
          }
      }
      for (int rc = 0; rc < 160; ++rc)
      {
          for (int i = 0; i < 9*9; i++){
              l_input[i] = input0[9*9*rc+i];
       }
              #pragma unroll 4
          for (int yy = 0; yy < 7; ++yy)
          {
              #pragma unroll
              for (int xx = 0; xx < 7; ++xx)
              {
                  float temp_0 = 0;
                  #pragma unroll
                  for (int rx = 0; rx < 3; ++rx)
                  {
                      temp_0 += l_input[(yy+0) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
                  }
                  temp_out[yy][xx] += temp_0;
                  float temp_1 = 0;
                  #pragma unroll
                  for (int rx = 0; rx < 3; ++rx)
                  {
                      temp_1 += l_input[(yy+1) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
                  }
                  temp_out[yy][xx] += temp_1;
                  float temp_2 = 0;
                  #pragma unroll
                  for (int rx = 0; rx < 3; ++rx)
                  {
                      temp_2 += l_input[(yy+2) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
                  }
                  temp_out[yy][xx] += temp_2;
              }
          }
      }
      #pragma loop_coalesce
      for (int yy = 0; yy < 7; ++yy)
      {
          for (int xx = 0; xx < 7; ++xx)
          {
              temp_out[yy][xx] += input_bias[ff];
              temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
              write_channel_intel(conv2_2_5b_out_b1_channel, temp_out[yy][xx]);
          }
      }
  }
}



__kernel void Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float *restrict input1, constant float *restrict input2)
{
    float input0[832 * 7 * 7];
    
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5a_out_channel3);
    }
    //Local memory for Biases:
    __local  float input_bias[32];
  
    for(int b = 0; b < 32; b++){
        input_bias[b] = input2[b];
    }

    float l_input[49];
       for (int ff = 0; ff < 32; ++ff)
    {
        //Local weights 
        float input_weights[832];
		#pragma unroll 8
        for(int m = 0 ; m < 832 ;m++){
            input_weights[m] = input1[((ff * 832) + m)];
        }
        
        float temp_out[7][7];
        #pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 832; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            
#pragma unroll 4
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[ yy * 14 + xx] * input_weights[rc]);
                }
             
                
            }
        }
        #pragma loop_coalesce
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff]; 
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                write_channel_intel(conv3_1_5b_out_b2_channel, temp_out[yy][xx]);
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
                                                   constant float *restrict input2)
{
  float input0[2592];
 
  for (int i = 0; i < 2592; i++)
  {
      input0[i] = read_channel_intel(padding_5b_out_b2_channel);
  }
  __local  float input_bias[128];

  for(int b = 0; b < 128; b++){
      input_bias[b] = input2[b];
  }
  float l_input[9*9];
  for (int ff = 0; ff < 128; ++ff)
  {
      float input_weights[3*3*32];
    #pragma unroll 8
    for (int k = 0; k < 3*3*32; k++){
          input_weights[k] = input1[((ff *3*3*32) + k)];
   }
   float temp_out[7][7];
      #pragma loop_coalesce
      for (int l = 0; l < 7; l++ ){
          for (int j = 0; j < 7; j++){
              temp_out[l][j] = 0.0;
          }
      }
      for (int rc = 0; rc < 32; ++rc)
      {
          for (int i = 0; i < 9*9; i++){
              l_input[i] = input0[9*9*rc+i];
       }
              #pragma unroll 4
          for (int yy = 0; yy < 7; ++yy)
          {
              #pragma unroll
              for (int xx = 0; xx < 7; ++xx)
              {
                  float temp_0 = 0;
                  #pragma unroll
                  for (int rx = 0; rx < 3; ++rx)
                  {
                      temp_0 += l_input[(yy+0) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 0) * 3) + rx)];
                  }
                  temp_out[yy][xx] += temp_0;
                  float temp_1 = 0;
                  #pragma unroll
                  for (int rx = 0; rx < 3; ++rx)
                  {
                      temp_1 += l_input[(yy+1) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 1) * 3) + rx)];
                  }
                  temp_out[yy][xx] += temp_1;
                  float temp_2 = 0;
                  #pragma unroll
                  for (int rx = 0; rx < 3; ++rx)
                  {
                      temp_2 += l_input[(yy+2) * 9 + xx + rx] * input_weights[(((((rc) * 3) + 2) * 3) + rx)];
                  }
                  temp_out[yy][xx] += temp_2;
              }
          }
      }
      #pragma loop_coalesce
      for (int yy = 0; yy < 7; ++yy)
      {
          for (int xx = 0; xx < 7; ++xx)
          {
              temp_out[yy][xx] += input_bias[ff];
              temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
              write_channel_intel(conv3_2_5b_out_b2_channel, temp_out[yy][xx]);
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


__kernel void Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float *restrict input1, constant float *restrict input2)
{
    float input0[832 * 7 * 7];
    for (int i = 0; i < 832 * 7 * 7; i++)
    {
        input0[i] = read_channel_intel(maxpool_5b_out_b3_channel);
    }
    //Local memory for Biases:
    __local  float input_bias[128];
    #pragma unroll 8
    for(int b = 0; b < 128; b++){
        input_bias[b] = input2[b];
    }

    float l_input[49];
       for (int ff = 0; ff < 128; ++ff)
    {
        //Local weights 
        float input_weights[832];
		#pragma unroll 8
        for(int m = 0 ; m < 832 ;m++){
            input_weights[m] = input1[((ff * 832) + m)];
        }
        
        float temp_out[7][7];
        #pragma loop_coalesce
        for (int l = 0; l < 14; l++ ){
            for (int j = 0; j < 14; j++){
                temp_out[l][j] = 0;
            }
        }
        for (int rc = 0; rc < 832; rc++)
        {
            for (int i = 0; i < 7*7; i++){
                l_input[i] = input0[7*7*rc+i];
            }
            
#pragma unroll 4
            for (int yy = 0; yy < 7; ++yy)
            {
#pragma unroll
                for (int xx = 0; xx < 7; ++xx)
                {
                    temp_out[yy][xx] += (l_input[ yy * 14 + xx] * input_weights[rc]);
                }
             
                
            }
        }
        #pragma loop_coalesce
        for (int yy = 0; yy < 7; ++yy)
        {
            for (int xx = 0; xx < 7; ++xx)
            {
                temp_out[yy][xx] += input_bias[ff]; 
                temp_out[yy][xx] = (temp_out[yy][xx] > 0) ? temp_out[yy][xx] : 0.000000e+00f;
                write_channel_intel(conv4_1_5b_out_b3_channel, temp_out[yy][xx]);
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
