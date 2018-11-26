//ACL Kernel

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[11];

	//Initialize the array with 0
	#pragma unroll
	for(int i=0;i<11;i++)
		sum_copies[i]=0.0;
	#pragma unroll 16
	for (unsigned i = 0; i < vectorSize; i++)
	{
		double cur = (input[i]*0.5) + sum_copies[10];

		//shift register
		#pragma unroll
		for(int j=10;j>0;j--)
			sum_copies[j]=sum_copies[j-1];

		sum_copies[0]=cur;
	}

	#pragma unroll
	for(int i=0;i<11;i++)
		result = result + sum_copies[i];

	*output = result;
}
