#pragma OPENCL EXTENSION cl_intel_channels : enable
typedef struct ch_buf
{
float temp_buf[8];
}ch;
channel ch chan_in2 __attribute__((depth(8)));
__kernel void SimpleKernel(__global const float * restrict in,__global float * restrict out,const uint N)
{
	//Perform the Math Operation
	int i,j;
	struct ch_buf cb;
	for (i=0;i<N;i++)
	{
	cb=read_channel_intel(chan_in2);	
		for (j=0;j<8;j++)
		{
		out[i]=in[i]*cb.temp_buf[j];
		}
	 } 
}
__kernel void InputKernel(__global const float * restrict in2,const uint N)
{
	int i,j;
	struct ch_buf cb;
	for (i=0;i<N;i++)
	{
		for (j=0;j<8;j++)
		{
		cb.temp_buf[j]=in2[i];
		}	
	write_channel_intel(chan_in2,cb);
	}
}
