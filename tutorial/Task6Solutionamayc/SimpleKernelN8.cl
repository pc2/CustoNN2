//ACL Kernel
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float chan_in2 __attribute__((depth(8)));

__kernel 
void SimpleKernelN8(__global const float * restrict in, __global const float * restrict in2, __global float * restrict out, uint N)
{
	//Perform the Math Operation

	for (uint index = 0; index < N; index++){
	  out[index] = in[index] * read_channel_intel(chan_in2);
	//mem_fence(CLK_CHANNEL_MEM_FENCE);	
}
}

__kernel 
void InputKernel(__global const float * restrict in2, uint N)
{
	for (uint index = 0; index < N; index++){
	  write_channel_intel(chan_in2, in2[index]);
	//mem_fence(CLK_CHANNEL_MEM_FENCE);
}
}
