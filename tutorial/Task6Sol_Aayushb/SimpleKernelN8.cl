//ACL Kernel
#pragma OPENCL EXTENSION cl_intel_channels : enable channel
channel float chan_in2;
__kernel void SimpleKernel(__global const float * restrict in, __global float * restrict out, uint N)
{
	#pragma unroll 8
	for (uint index = 0; index < N; index++)
		out[index] = in[index] * read_channel_intel(chan_in2);
	
}

__kernel void InputKernel(__global const float * restrict in2, uint N)
{
	#pragma unroll 8
	for (uint index = 0; index < N; index++)
		write_channel_intel(chan_in2, in2[index]);	
 	
}

