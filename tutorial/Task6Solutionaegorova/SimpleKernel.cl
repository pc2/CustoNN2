//ACL Kernel
#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float chan_in2;

__kernel 
void SimpleKernel(__global const float * restrict in, __global float * restrict out, uint N)
{
	//Perform the Math Operation
	for (uint index = 0; index < N; index++)
	  out[index] = in[index] * read_channel_intel(chan_in2);
}

void InputKernel(__global const float * restrict in2, uint N) 
{
	for (uint index = 0; index < N; index++)
	  write_channel_intel(chan_in2, in[index] * in2[index]);
}
