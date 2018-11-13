//ACL Kernel

 __kernel void SimpleKernel (global float *a, __global float *b, __global float *c) 
{
	int i;
	i=get_global_id(0);
	c[i] = a[i] + b[i];
		
}
