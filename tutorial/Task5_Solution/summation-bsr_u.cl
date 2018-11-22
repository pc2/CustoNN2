//ACL Kernel

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[20];
	#pragma unroll 
	for (unsigned i = 0; i<20; i++)
	{ 
		sum_copies[i] = 0;
	}

	#pragma unroll 16
	for (unsigned i = 0; i < vectorSize; i++)
	{
		
		double cur = (input[i]*0.5 + sum_copies[19]);
		
		for (unsigned k = 19; k>0; k--)
		{
			sum_copies[k]=sum_copies[k-1];

	 	}
	
	sum_copies[0]=cur;
	
	}
	#pragma unroll 
	for (unsigned j=0; j<20; j++)
	{
		result+=sum_copies[j];

	}
		
	*output = result;
}
