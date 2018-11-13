//ACL Kernel

 __kernel void SimpleKernel ( __global float *a, __global float *b, __global float *c, int N) 
{
	int i;
	for (i = 0; i < N; i++)
		c[i] = a[i] + b[i];
}
