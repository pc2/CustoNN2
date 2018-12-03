//ACL Kernel

/*
 *
 *Source for this code: https://www.intel.com/content/www/us/en/programmable/documentation/mwh1391807965224/ewa1455918581924/ewa1456417646847/ewa1456430907569.html
 * OpenCL Programming Guide
 *
 */
#define size 8
//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//declare a channel - file scope
//Widening the channel
channel float chan_in2[size+1];
__attribute__((max_global_work_dim(0)))
__kernel void SimpleKernel(__global const float * restrict in, __global float * restrict out, uint N)
{

	//Perform the Math Operation
	for (uint index = 0; index < N; index++)
	  out[index] = in[index] * read_channel_intel(chan_in2[size]);  //Reads the channel
}
__attribute__((max_global_work_dim(0)))
//Kernel for writing the data into a channel
__kernel void InputKernel(__global const float * restrict in2,uint N){
	for(uint i=0;i<N;i++)
		write_channel_intel(chan_in2[0],in2[i]); //Sending data across channel
}
//replicating kernels , as 8 compute elements
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(size)))
__kernel void plusOne() {
	int compute_id = get_compute_id(0);
	float input = read_channel_intel(chan_in2[compute_id]);
	write_channel_intel(chan_in2[compute_id+1], input);
}
