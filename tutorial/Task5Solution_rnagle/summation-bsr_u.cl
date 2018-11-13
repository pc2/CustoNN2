//ACL Kernel
#define II 180

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	
	double sum_copies[II+1];
	for(int j=0;j<II+1;j++)
	{
		sum_copies[j]=0;
	}
	#pragma unroll 16	
	for (unsigned i = 0; i < vectorSize; i++)
	{
		
		double curr=(input[i]*0.5) + sum_copies[II];
		#pragma unroll
		for(int j=II;j>0;j--)
		{
			sum_copies[j]=sum_copies[j-1];
		}
		sum_copies[0]=curr;
		
	}
	#pragma unroll
	for(int j=0;j<=II;j++)
	{
		result+=sum_copies[j];
	}
	*output = result;
}
