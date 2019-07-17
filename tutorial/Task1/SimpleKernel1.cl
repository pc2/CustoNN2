//ACL Kernel
__kernel void SimpleKernel ( __global float *restrict X,__global float *restrict Y,__global float *restrict Z)
{
size_t i = get_global_id(0);
Z[i] = X[i] + Y[i];
}

