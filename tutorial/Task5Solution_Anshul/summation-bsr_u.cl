//ACL Kernel

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[180];
	#pragma unroll
	for (int j=0;j<180;j++)
	{
		sum_copies[j]=0;
	}
	#pragma unroll 16
	for (unsigned i = 0; i < vectorSize; i++)
	{
	double curr;
	//result = (input[i]*0.5) + result;
	curr = (input[i]*0.5) + sum_copies[179];
		#pragma unroll
		for ( int k=180;k>0;k--)
		{
			sum_copies[k]=sum_copies[k-1];
		}
	sum_copies[0] = curr;
	}
	#pragma unroll
	for( int l=0;l<180;l++)
	{
		result += sum_copies[l];
	}
	*output = result;
}
