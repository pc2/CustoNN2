#pragma OPENCL EXTENSION cl_intel_channels : enable


typedef struct chan_buf
{
float temp[8];
}c1;


channel c1 chan_in2 __attribute__((depth(8)));

__kernel void SimpleKernel ( __global const float *restrict Buffer_In, __global float *restrict Buffer_Out, const uint vectorSize)
{
int i=0;
bool success;




while(i<vectorSize)
{
struct chan_buf c2=read_channel_intel(chan_in2);

for(int j=0;j<8;j++,i++)
Buffer_Out[i]=Buffer_In[i]*c2.temp[j];
}


}

__kernel void InputKernel ( __global const float *restrict Buffer_In2,const uint vectorSize)
{
int i=0;


while(i<vectorSize)
{
struct chan_buf c2;
for(int j=0;j<8;j++,i++)
{
c2.temp[j]=Buffer_In2[i];
}
write_channel_intel(chan_in2,c2);
}
}