//Enable the channel extension
 #pragma OPENCL EXTENSION cl_intel_channels : enable

typedef struct IO_buffer {
        float temp_buffer[8];
}iob;


// IO input channel
channel iob IO_input __attribute__((depth(2)))
                           __attribute__((io("kernel_input_ch0"))); 

//output from maxpool
channel float maxOutChannel1 __attribute__((depth(16)));
channel float maxOutChannel2 __attribute__((depth(16)));
channel float maxOutChannel3 __attribute__((depth(16)));
channel float maxOutChannel4 __attribute__((depth(16)));

//first output - 64 
channel float conv1OutChannel __attribute__((depth(16)));

//second output - 128
channel float conv2_1OutChannel __attribute__((depth(16)));
channel float conv2_2OutChannel __attribute__((depth(16)));

//third output - 32
channel float conv3_1OutChannel __attribute__((depth(16)));
channel float conv3_2OutChannel __attribute__((depth(16)));

//forth output - 32
channel float padding4_1OutChannel __attribute__((depth(16)));
channel float max4_2OutChannel __attribute__((depth(16)));
channel float conv4_3OutChannel __attribute__((depth(16)));

// IO output channel
channel iob IO_output __attribute__((depth(2)))
                           __attribute__((io("kernel_output_ch0"))); 


__kernel void MaxPool_3a_3x3_MaxPool() {
  float input0[192*28*28];
  int index = 0;
  for (int i = 0; i < 192*28*28/8; i++){
  	struct IO_buffer temp_iob;
  	temp_iob = read_channel_intel(IO_input);
  	for (int j = 0; j < 8; j++){
  		input0[index] = temp_iob.temp_buffer[j];
  		index++;	
  	}
  }
  float tensor[192 * 28 * 28];
  for (int ax1 = 0; ax1 < 192; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((ax2 * 2) < (56 - rv)) && ((ax3 * 2) < (56 - rv1))) ? input0[((((((((ax1 * 28) + ax2) * 2) + rv) * 28) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      write_channel_intel(maxOutChannel1, tensor[((((ax1 * 28) + ax2) * 28) + ax3)]);
      write_channel_intel(maxOutChannel2, tensor[((((ax1 * 28) + ax2) * 28) + ax3)]);
      write_channel_intel(maxOutChannel3, tensor[((((ax1 * 28) + ax2) * 28) + ax3)]);
      write_channel_intel(maxOutChannel4, tensor[((((ax1 * 28) + ax2) * 28) + ax3)]);
      }
    }
  }
}

__kernel void Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict input1, __global float* restrict input2) {
  float input0[192*28*28];
  for (int i = 0; i < 192*28*28; i++){
        input0[i] = read_channel_intel(maxOutChannel1);
  }
  float temp_0;
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        temp_0 = input2[ff];
        for (int rc = 0; rc < 192; ++rc) {
          temp_0 += input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)];
        }
        temp_0 = ( temp_0 > 0) ? temp_0 : 0.000000e+00f;
        write_channel_intel(conv1OutChannel, temp_0);
      }
    }
  }
}

__kernel void Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict input1 , __global float* restrict input2) {
  float input0[192*28*28];
  for (int i = 0; i < 192*28*28; i++){
        input0[i] = read_channel_intel(maxOutChannel2);
      }
  float temp_0;
  for (int ff = 0; ff < 96; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        temp_0 = input2[ff];
        for (int rc = 0; rc < 192; ++rc) {
          temp_0 += input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)];
        }
        temp_0 = ( temp_0 > 0) ? temp_0 : 0.000000e+00f;
        write_channel_intel(conv2_1OutChannel, temp_0);
      }
    }
  }
}


__kernel void Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict weights, 
                                                    __global float * restrict bias){
  
  float img[96*28*28];
  for (int i = 0; i < 96*28*28; i++){
    img[i] = read_channel_intel(conv2_1OutChannel);
  }

  int i,j,k,t;
  int temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  image_size = 28*28;
  for (layer = 0; layer <128; layer++) {
    float temp_conv_val = bias[layer];	
      for (i = 0; i < 28; i+=1){
        for (j = 0; j < 28; j+=1) {
	  for(int d=0;d<96;d++){
      
          int PaddedX = i;
          int PaddedY = j;
          int paderX = i - 1;
          int paderY = j - 1;

           for(int filterX=0; filterX<3; filterX++){
              for(int filterY=0; filterY<3; filterY++){
                if(paderX<0||paderX>=28||paderY<0||paderY>=28){}else{
                  temp_conv_val  += img[(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*96)+(d*3*3)+(filterX*3)+filterY] ;
                  PaddedY++;
                }
                paderY++;
              }
              PaddedX++;
              paderX++;
              PaddedY=j;
              paderY = j - 1;
          }  
	}
        temp_conv_val = (temp_conv_val>0) ? temp_conv_val : 0;
        write_channel_intel(conv2_2OutChannel, temp_conv_val);
      }
    }
  }
}

__kernel void Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict input1, __global float* restrict input2) {
  float input0[192*28*28];
  for (int i = 0; i < 192*28*28; i++){
        input0[i] = read_channel_intel(maxOutChannel3);
      }
  float temp_0;
  for (int ff = 0; ff < 16; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        temp_0 = input2[ff];
        for (int rc = 0; rc < 192; ++rc) {
          temp_0 += input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)];
        }
        temp_0 = ( temp_0 > 0) ? temp_0 : 0.000000e+00f;
        write_channel_intel(conv3_1OutChannel, temp_0);
      }
    }
  }
}

__kernel void Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float * restrict weights, 
                                                      __global float * restrict bias){
  
  float img[16*28*28];
  for (int i = 0; i < 16*28*28; i++){
    img[i] = read_channel_intel(conv3_1OutChannel);
  }
  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  index = 0;
  image_size = 28*28;
  for (layer = 0; layer <32; layer++){
      double temp_conv_val = bias[layer];
      for (i = 0; i < 28; i+=1){
        for (j = 0; j < 28; j+=1){
    	  for(int d=0;d<16;d++){      
          int PaddedX = i;
          int PaddedY = j;
          int paderX = i - 1;
          int paderY = j - 1;

          for(int filterX=0; filterX<3; filterX++){
              for(int filterY=0; filterY<3; filterY++){
                if(paderX<0||paderX>=28||paderY<0||paderY>=28){}else{
                  temp_conv_val  += img[(d*28*28)+(28*PaddedX)+PaddedY] *weights[(layer*3*3*16)+(d*3*3)+(filterX*3)+filterY] ;
                  PaddedY++;
                }
                paderY++;
              }
              PaddedX++;
              paderX++;
              PaddedY=j;
              paderY = j - 1;
          }
	}
	temp_conv_val = (temp_conv_val>0) ? temp_conv_val : 0;
	write_channel_intel(conv3_2OutChannel, temp_conv_val);
      }
    }
  }
}

__kernel void Padding_Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool() {
  float input0[192*28*28];
  for (int i = 0; i < 192*28*28; i++){
    input0[i] = read_channel_intel(maxOutChannel4);
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    write_channel_intel(padding4_1OutChannel,input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 192) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))]);
  }
}

__kernel void Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool() {
  float input0[192*28*28], tensor[192*28*28];
  for (int i = 0; i < 192*28*28; i++){
    input0[i] = read_channel_intel(padding4_1OutChannel);
  }
  for (int ax1 = 0; ax1 < 192; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? input0[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
          }
        }
      }
    }
  }
  for (int i = 0; i < 192*28*28; i++){
    write_channel_intel(max4_2OutChannel, tensor[0]);
  }
}


__kernel void Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict input1 , __global float* restrict input2) {
  float input0[192*28*28];
  for (int i = 0; i < 192*28*28; i++){
    input0[i] = read_channel_intel(max4_2OutChannel);
  }
    float temp_0;
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        temp_0 = input2[ff];
        for (int rc = 0; rc < 192; ++rc) {
          temp_0 += input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)];
        }
        temp_0 = ( temp_0 > 0) ? temp_0 : 0.000000e+00f;
        write_channel_intel(conv4_3OutChannel, temp_0);
      }
    }
  }
}

__kernel void Mixed_3b_concat() {
  float input0[64*28*28], input1[128*28*28], input2[32*28*28], input3[32*28*28], output[200704];
  for (int i = 0; i < 64*28*28; i++){
    input0[i] = read_channel_intel(conv1OutChannel);
  }
  for (int i = 0; i < 128*28*28; i++){
    input1[i] = read_channel_intel(conv2_2OutChannel);
  }
  for (int i = 0; i < 32*28*28; i++){
    input2[i] = read_channel_intel(conv3_2OutChannel);
    input3[i] = read_channel_intel(conv4_3OutChannel);
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    output[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(224 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -175616)] : (float)(192 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) ? input1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -150528)] : (float)(64 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) ? input2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -50176)] : input3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256))])));
  }


  int index = 0;
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ax0_ax1_fused_ax2_fused_ax3_fused_inner += 8) {
  	struct IO_buffer temp_iob;
  	for (int j = 0; j < 8; j++){
  		temp_iob.temp_buffer[j] = output[index];
      index++;	
  	}
	write_channel_intel(IO_output, temp_iob);
  }

}

