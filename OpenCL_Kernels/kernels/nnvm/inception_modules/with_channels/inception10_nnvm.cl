//Enable the channel extension
 #pragma OPENCL EXTENSION cl_intel_channels : enable

typedef struct IO_buffer {
        float temp_buffer[8];
}iob;

//first output
channel float conv1OutChannel __attribute__((depth(16)));

//second output
channel float max2OutChannel __attribute__((depth(16)));

//third output
channel float conv3OutChannel __attribute__((depth(16)));

//fourth output
channel float conv4OutChannel __attribute__((depth(16)));

// IO output channel
channel iob IO_output __attribute__((depth(0)))
                           __attribute__((io("kernel_output_ch0"))); 

__kernel void Conv2d_1a_7x7_Conv2D(__global unsigned char * restrict img, 
        __global float * restrict weights, 
        __global float * restrict bias){

  int i,j,k,t;
  int temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  image_size = 224 * 224;
    for (layer = 0; layer <64; layer++){
	float temp_conv_val = bias[layer];
      for (i = 0; i < 224; i+=2){
        for (j = 0; j < 224; j+=2){
        	for(int d=0;d<3;d++){

            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 2;
            int paderY = j - 2;

             for(int filterX=0; filterX<7; filterX++){
                for(int filterY=0; filterY<7; filterY++){
                  if(paderX<0||paderX>=224||paderY<0||paderY>=224){}else{
                    temp_conv_val  += img[(d*224*224)+(224*PaddedX)+PaddedY] *weights[(layer*7*7*3)+(d*7*7)+(filterX*7)+filterY] ;
                    PaddedY++;
                }
                paderY++;
             }
              PaddedX++;
              paderX++;
              PaddedY=j;
              paderY = j - 2;
            }
          temp_conv_val = (temp_conv_val>0) ? temp_conv_val : 0;
          write_channel_intel(conv1OutChannel, temp_conv_val);
        }
      }
    }
  }
}


__kernel void MaxPool_2a_3x3_MaxPool(){
  float input0[64*112*112], tensor[64*56*56];
  for (int i = 0; i < 64*112*112; i++){
    input0[i] = read_channel_intel(conv1OutChannel);
  }
  for (int ax1 = 0; ax1 < 64; ++ax1) {
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = max(tensor[((((ax1 * 56) + ax2) * 56) + ax3)], (float)((((ax2 * 2) < (112 - rv)) && ((ax3 * 2) < (112 - rv1))) ? input0[((((((((ax1 * 56) + ax2) * 2) + rv) * 56) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      }
    }
  }
  for (int i = 0; i < 64*56*56; i++){
    write_channel_intel(max2OutChannel, tensor[i]);
  }
}

__kernel void Conv2d_2b_1x1_Conv2D(__global float* restrict input1, __global float* restrict input2) {
  float input0[64*56*56];
  for (int i = 0; i < 64*56*56; i++){
    input0[i] = read_channel_intel(max2OutChannel);
  }
  float temp_0;
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        temp_0 = input2[ff] ;
        for (int rc = 0; rc < 64; ++rc) {
          temp_0 += (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 64) + rc)]);
        }
        temp_0 = ( temp_0 > 0) ? temp_0 : 0.000000e+00f;
        write_channel_intel(conv3OutChannel, temp_0);
      }
    }
  }
}


__kernel void Conv2d_2c_3x3_Conv2D(__global float * restrict weights, 
                                  __global float * restrict bias){
  float img[64*56*56];
  for (int i = 0; i < 64*56*56; i++){
    img[i] = read_channel_intel(conv3OutChannel);
  }
  int i,j,k,t;
  int temp_index, filter_index, image_index;
  int layer, current_layer, image_size;
  image_size = 56*56;
    for (layer = 0; layer <192; layer++){
float temp_conv_val = bias[layer];
      for (i = 0; i < 56; i+=1) {
        for (j = 0; j < 56; j+=1) {
        	for(int d=0;d<64;d++){
            int PaddedX = i;
            int PaddedY = j;
            int paderX = i - 1;
            int paderY = j - 1;

             for(int filterX=0; filterX<3; filterX++){
                for(int filterY=0; filterY<3; filterY++){
                  if(paderX<0||paderX>=56||paderY<0||paderY>=56){}else{
                      temp_conv_val  += img[(d*56*56)+(56*PaddedX)+PaddedY] *weights[(layer*3*3*64)+(d*3*3)+(filterX*3)+filterY] ;
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
          write_channel_intel(conv4OutChannel, temp_conv_val);
          }
        }
      
    }
}

__kernel void MaxPool_3a_3x3_MaxPool() {
  float input0[192*56*56], tensor[192*28*28];
  for (int i = 0; i < 192*56*56; i++){
    input0[i] = read_channel_intel(conv4OutChannel);
  }
  for (int ax1 = 0; ax1 < 192; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((ax2 * 2) < (56 - rv)) && ((ax3 * 2) < (56 - rv1))) ? input0[((((((((ax1 * 28) + ax2) * 2) + rv) * 28) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      }
    }
  }

  int index = 0;
  for (int i = 0; i < 64*56*56; i+=8){
    struct IO_buffer temp_iob;
    for (int j = 0; j < 8; j++){
      temp_iob.temp_buffer[j] = tensor[index];
      index++;  
    }
    write_channel_intel(IO_output, temp_iob);
  }

}

