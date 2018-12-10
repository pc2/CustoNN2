//ACL Kernel
#pragma OPENCL EXTENSION cl_intel_channels : enable

typedef struct channels{
	float ch_data[8];
}ch;

channel ch chan_in2 __attribute__((depth(8)));
__kernel
void SimpleKernel(__global const float * restrict in, __global float * restrict out, uint N)
{
	//in2 = read_channel_intel(chan_in2);
	//Perform the Math Operation
	for (uint i=0;i<N;){
		struct channels read_ch=read_channel_intel(chan_in2);
		for (uint index=0;index<8;index++,i++){
			out[i] = in[i] * read_ch.ch_data[index];
		}
	 } 
}

__kernel void InputKernel(__global const float * restrict in2, uint N)
{

	for (uint i=0;i<N;){
		struct channels read_ch;
		for (uint index = 0; index < 8; index++,i++){
			read_ch.ch_data[index]=in2[i];
		}
		write_channel_intel(chan_in2,read_ch);
	}
	
}

