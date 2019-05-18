#pragma OPENCL EXTENSION cl_intel_channels : enable
typedef struct cbf // Creation of channel buffer using struct
{
float bf[8];
}ch;		// datatype ch
channel ch chan_in2 __attribute__((depth(8))); // using channel with type ch
__kernel void InputKernel(__global const float * restrict in2,const uint N)
{
int i,j;
struct cbf chb;
for (i=0;i<N;i++)   // vectorsize
{
for (j=0;j<8;j++)  // 8 operations
{
chb.bf[j]=in2[i];
}	
write_channel_intel(chan_in2,chb); // write buffer to a channel
}
}

__kernel void SimpleKernel(__global const float * restrict in,__global float * restrict out,const uint N)
{
int i,j;
struct cbf chb;
for (i=0;i<N;i++)  // vectorsize
{	
chb=read_channel_intel(chan_in2);	// first reading from the channel to buffer 	
for (j=0;j<8;j++)
{
out[i]=in[i]*chb.bf[j];		// after reading from channel , then using buffer for multiply operation
}
} 
}

