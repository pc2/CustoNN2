//ACL Kernel 
__kernel void SimpleKernel(__global float *restrict X,__global float *restrict Y,__global float *restrict Z, int N)
{
int index;
for(index = 0;index < N;index++)
Z[index]=X[index] + Y[index];
}
