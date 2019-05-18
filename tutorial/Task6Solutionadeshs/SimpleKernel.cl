//ACL Kernel
//Enable the channel extension
#pragma OPENCL EXTENSION cl_intel_channels : enable

//declare a channel - file scope
channel float chan_in2;

__kernel void SimpleKernel(__global const float * restrict in, __global float * restrict out, uint N)
{

	//Perform the Math Operation
	for (uint index = 0; index < N; index++)
	  out[index] = in[index] * read_channel_intel(chan_in2);  //Reads the channel
}

//Kernel for writing the data into a channel
__kernel void InputKernel(__global const float * restrict in2,uint N){
	for(uint i=0;i<N;i++)
		write_channel_intel(chan_in2,in2[i]); //Sending data across channel
}
