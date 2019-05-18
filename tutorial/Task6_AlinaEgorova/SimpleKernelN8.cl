//ACL Kernel
#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float8 chan_in2 __attribute__((depth(0)));

__kernel 
void SimpleKernelN8(__global const float * restrict in, __global float * restrict out, uint N)
{
	__local float8 f_chain_in2;
	//Perform the Math Operation
	for (uint index = 0; index < N; index+=8) {
		f_chain_in2 = read_channel_intel(chan_in2);

		out[index] = in[index] * f_chain_in2.s0;
		out[index+1] = in[index+1] * f_chain_in2.s1;
		out[index+2] = in[index+2] * f_chain_in2.s2;
		out[index+3] = in[index+3] * f_chain_in2.s3;
		out[index+4] = in[index+4] * f_chain_in2.s4;
		out[index+5] = in[index+5] * f_chain_in2.s5;
		out[index+6] = in[index+6] * f_chain_in2.s6;
		out[index+7] = in[index+7] * f_chain_in2.s7;
	}
}


__kernel 
void InputKernel(__global const float * restrict in2, uint N) 
{
	for (uint index = 0; index < N; index+=8)
		write_channel_intel(chan_in2, (float8)(in2[index], 
							in2[index+1], 
							in2[index+2],
							in2[index+3],
							in2[index+4],
							in2[index+5],
							in2[index+6],
							in2[index+7])
					);
}
