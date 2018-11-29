//ACL Kernel

#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float chan_in2;

__kernel void SimpleKernel(__global const float * restrict in, __global float * restrict out,const uint N)
{
	//Perform the Math Operation
	int i;
	#pragma unroll 8
	for (i = 0; i < N; i++)

	  out[i] = in[i] * read_channel_intel(chan_in2);;
	  

}



__kernel void InputKernel(__global const float * restrict in2,const uint N)
{
	int i;
	#pragma unroll 8
	for ( i = 0; i < N; i++)
	{
	write_channel_intel(chan_in2,in2[i]);
		
	}
}
