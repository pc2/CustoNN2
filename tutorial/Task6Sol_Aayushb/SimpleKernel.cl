//ACL Kernel
#pragma OPENCL EXTENSION cl_intel_channels : enable channel
channel float chan_in2 __attribute__((depth(0)));
__kernel void SimpleKernel(__global const float * restrict in, __global float * restrict out, int vectorSize)
{
	
	for (uint index = 0; index < vectorSize; index++)
		out[index] = in[index] * read_channel_intel(chan_in2);
	
}

__kernel void InputKernel(__global const float * restrict in2, int vectorSize)
{
	for (uint index = 0; index < vectorSize; index++)
		write_channel_intel(chan_in2, in2[index]);	
 	
}

