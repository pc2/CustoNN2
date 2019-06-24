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

//first output - 256
channel float conv1OutChannel __attribute__((depth(16)));

//second output - 320
channel float conv2_1OutChannel __attribute__((depth(16)));
channel float conv2_2OutChannel __attribute__((depth(16)));

//third output - 128
channel float conv3_1OutChannel __attribute__((depth(16)));
channel float conv3_2OutChannel __attribute__((depth(16)));

//fourth output - 128
channel float padding4_1OutChannel __attribute__((depth(16)));
channel float max4_2OutChannel __attribute__((depth(16)));
channel float conv4_3OutChannel __attribute__((depth(16)));

// IO output channel
channel iob IO_output __attribute__((depth(2)))
                           __attribute__((io("kernel_output_ch0"))); 

__kernel void MaxPool_5a_2x2_MaxPool(){
float input0[832*7*7];
int index=0;
for (int i = 0; i < 832*7*7/8; i++){
	struct IO_buffer temp_iob;
	temp_iob = read_channel_intel(IO_input);
  	for (int j = 0; j < 8; j++){
  		input0[index] = temp_iob.temp_buffer[j];
  		index++;	
	}
	
}

float tensor[832*7*7];
for (int ax1 = 0; ax1 < 832; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 2; ++rv) {
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], input0[((((((((ax1 * 7) + ax2) * 2) + rv) * 7) + ax3) * 2) + rv1)]);
          }
        }

      write_channel_intel(maxOutChannel1, tensor[((((ax1 * 7) + ax2) * 7) + ax3)]);
      write_channel_intel(maxOutChannel2, tensor[((((ax1 * 7) + ax2) * 7) + ax3)]);
      write_channel_intel(maxOutChannel3, tensor[((((ax1 * 7) + ax2) * 7) + ax3)]);
      write_channel_intel(maxOutChannel4, tensor[((((ax1 * 7) + ax2) * 7) + ax3)]);
      }
    }
  }
}

__kernel void Mixed_5b_Branch_0_Conv2d_0a_1x1_Conv2D(
                __global float* restrict input1, 
                __global float* restrict input2) {
 float input0[832*7*7];
  for (int i = 0; i < 832*7*7; i++){
        input0[i] = read_channel_intel(maxOutChannel1);
  }
  float temp_0;
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
	temp_0 = input2[ff];
	for (int rc = 0; rc < 832; ++rc) {
	temp_0 += input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)];
	}
	temp_0 = ( temp_0 > 0) ? temp_0 : 0.000000e+00f;
        write_channel_intel(conv1OutChannel, temp_0);
      }
    }
  }
}

__kernel void Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict input1, 
                __global float* restrict input2) {
float input0[832*7*7];
  for (int i = 0; i < 832*7*7; i++){
        input0[i] = read_channel_intel(maxOutChannel2);
      }
float temp_0;
for (int ff = 0; ff < 160; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
temp_0 = input2[ff];
for (int rc = 0; rc < 832; ++rc) {
temp_0 += input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)];
}
temp_0 = ( temp_0 > 0) ? temp_0 : 0.000000e+00f;
        write_channel_intel(conv2_1OutChannel, temp_0);
      }
    }
  }
}
 __kernel void Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float * restrict weights, 
                                                    __global float * restrict bias){

float img[160*7*7];
  for (int i = 0; i < 160*7*7; i++){
    img[i] = read_channel_intel(conv2_1OutChannel);
  }
int i,j,k,t;
  int temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  image_size = 7*7;

for (layer = 0; layer <320; layer++) {
    float temp_conv_val = bias[layer];	
      for (i = 0; i < 7; i+=1){
        for (j = 0; j < 7; j+=1) {
	  for(int d=0;d<160;d++){
      
          int PaddedX = i;
          int PaddedY = j;
          int paderX = i - 1;
          int paderY = j - 1;

           for(int filterX=0; filterX<3; filterX++){
              for(int filterY=0; filterY<3; filterY++){
                if(paderX<0||paderX>=7||paderY<0||paderY>=7){}else{
                  temp_conv_val  += img[(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*3*3*160)+(d*3*3)+(filterX*3)+filterY] ;
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

__kernel void Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict input1, __global float* restrict input2) {
float input0[832*7*7];
  for (int i = 0; i < 832*7*7; i++){
        input0[i] = read_channel_intel(maxOutChannel3);
      }
float temp_0;
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        temp_0 = input2[ff];
        for (int rc = 0; rc < 832; ++rc) {
          temp_0 += input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)];
        }
        temp_0 = ( temp_0 > 0) ? temp_0 : 0.000000e+00f;
        write_channel_intel(conv3_1OutChannel, temp_0);
      }
    }
  }
}
__kernel void Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D(__global float * restrict weights, 
        						__global float * restrict bias){
float img[32*7*7];
  for (int i = 0; i < 32*7*7; i++){
    img[i] = read_channel_intel(conv3_1OutChannel);
  }
  int i,j,k,t;
  int index, temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  index = 0;
  image_size = 7*7;
  for (layer = 0; layer <128; layer++){
      double temp_conv_val = bias[layer];
      for (i = 0; i < 7; i+=1){
        for (j = 0; j < 7; j+=1){
    	  for(int d=0;d<32;d++){      
          int PaddedX = i;
          int PaddedY = j;
          int paderX = i - 1;
          int paderY = j - 1;

          for(int filterX=0; filterX<3; filterX++){
              for(int filterY=0; filterY<3; filterY++){
                if(paderX<0||paderX>=7||paderY<0||paderY>=7){}else{
                  temp_conv_val  += img[(d*7*7)+(7*PaddedX)+PaddedY] *weights[(layer*3*3*32)+(d*3*3)+(filterX*3)+filterY] ;
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
__kernel void Padding_Mixed_5b_Branch_3_MaxPool_0a_3x3_MaxPool() {
float input0[832*7*7];
  for (int i = 0; i < 832*7*7; i++){
    input0[i] = read_channel_intel(maxOutChannel4);
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    write_channel_intel(padding4_1OutChannel,input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 49) * 832) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49))]);
  }
}

__kernel void Mixed_5b_Branch_3_MaxPool_0a_3x3_MaxPool() {
float input0[832*7*7], tensor[832*7*7];
  for (int i = 0; i < 832*7*7; i++){
    input0[i] = read_channel_intel(padding4_1OutChannel);
  }
  for (int ax1 = 0; ax1 < 832; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (8 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (8 - rv1))) ? input0[(((((((ax1 * 7) + ax2) + rv) * 7) + ax3) + rv1) + -8)] : -3.402823e+38f));
          }
        }
      }
    }
  }
  for (int i = 0; i < 832*7*7; i++){
    write_channel_intel(max4_2OutChannel, tensor[0]);
  }
}
__kernel void Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict input1 , __global float* restrict input2) {
  float input0[832*7*7];
  for (int i = 0; i < 832*7*7; i++){
    input0[i] = read_channel_intel(max4_2OutChannel);
  }
    float temp_0;
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        temp_0 = input2[ff];
        for (int rc = 0; rc < 832; ++rc) {
          temp_0 += input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 832) + rc)];
        }
        temp_0 = ( temp_0 > 0) ? temp_0 : 0.000000e+00f;
        write_channel_intel(conv4_3OutChannel, temp_0);
      }
    }
  }
}

__kernel void Mixed_5b_concat() {
float input0[256*7*7], input1[320*7*7], input2[128*7*7], input3[128*7*7], output[832*7*7];
  for (int i = 0; i < 256*7*7; i++){
    input0[i] = read_channel_intel(conv1OutChannel);
  }
  for (int i = 0; i < 320*7*7; i++){
    input1[i] = read_channel_intel(conv2_2OutChannel);
  }
    for (int i = 0; i < 128*7*7; i++){
    input2[i] = read_channel_intel(conv3_2OutChannel);
    input3[i] = read_channel_intel(conv4_3OutChannel);
  }
for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    output[ax0_ax1_fused_ax2_fused_ax3_fused_inner]=(float)((704 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832)) + -34496)] : (float)((576 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? input1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832)) + -28224)] : (float)((256 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? input2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832)) + -12544)] : input3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832))])));
  }

int index = 0;
for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ax0_ax1_fused_ax2_fused_ax3_fused_inner+=8){
	struct IO_buffer temp_iob;
  	for (int j = 0; j < 8; j++){
		temp_iob.temp_buffer[j] = output[index];
      index++;	
  	}
	write_channel_intel(IO_output, temp_iob);
  }

}




















