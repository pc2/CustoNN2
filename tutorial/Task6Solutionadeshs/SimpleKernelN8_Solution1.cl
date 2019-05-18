//ACL Kernel
//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//declare a channel - file scope
// 32bits*8elements=256 bits width channel
channel float8 chan_in2;

__kernel void SimpleKernel(__global const float8 * restrict in, __global float8 * restrict out, uint N)
{
	//Perform the Math Operation
	for (uint index = 0; index < (N/8); index++){
		out[index]= in[index] * read_channel_intel(chan_in2); //Reads the channel
	}
}

//Kernel for writing the data into a channel
__kernel void InputKernel(__global const float8 * restrict in2,uint N){
	for(uint i=0; i<(N/8); i++) {
		write_channel_intel(chan_in2,in2[i]); //Sending 8 data elements across channel
	}
}
