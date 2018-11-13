//ACL Kernel
__kernel 
void SimpleKernel(__global const float * restrict in, __global const float * restrict in2, __global float * restrict out, uint N)
{
	//Perform the Math Operation
	for (uint index = 0; index < N; index++)
	  out[index] = in[index] * in2[index];
}