//ACL Kernel

#define ChanSize 8


#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float chan_in2[ChanSize+1] ;


__attribute__((max_global_work_dim(0)))
__kernel void InputKernel(__global const float * restrict in2, uint N)
{
	for (uint i = 0 ; i < N ; i++)
	{
	write_channel_intel (chan_in2[0], in2[i]);
	}


}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(ChanSize)))
__kernel void plusOne() 
{
 int compute_id = get_compute_id(0);
 float input = read_channel_intel(chan_in2[compute_id]);
 write_channel_intel(chan_in2[compute_id+1], input );
}


__attribute__((max_global_work_dim(0)))
__kernel void SimpleKernelN8(__global const float * restrict in,__global float * restrict out, uint N)
{
	//Perform the Math Operation
	for (uint index = 0; index < N; index++)
	  out[index] = in[index] * read_channel_intel(chan_in2[ChanSize]);
}



