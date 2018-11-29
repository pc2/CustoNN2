#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float chan_in2;

__kernel void SimpleKernel ( __global float *Buffer_In, __global float *Buffer_Out,int vectorSize)
{
int i;
for(i=0;i<vectorSize;i++)
{
//Buffer_Out[i]=Buffer_In[i]+Buffer_In2[i];
Buffer_Out[i]=Buffer_In[i]*read_channel_intel(chan_in2);
}
}

__kernel void InputKernel ( __global float *Buffer_In2, int vectorSize)
{
int i;
for(i=0;i<vectorSize;i++)
{
write_channel_intel(chan_in2,Buffer_In2[i]);
}
}