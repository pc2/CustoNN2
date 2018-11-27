//ACL Kernel

//Use 177 elements shift registers
#define M 177

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[M];

	//Initialize the array with 0
	#pragma unroll
	for(int i=0;i<M;i++)
		sum_copies[i]=0.0;
	#pragma unroll 16
	for (unsigned i = 0; i < vectorSize; i++)
	{
		double cur = (input[i]*0.5) + sum_copies[M-1];

		//shift register
		#pragma unroll
		for(int j=M-1;j>0;j--)
			sum_copies[j]=sum_copies[j-1];

		sum_copies[0]=cur;
	}

	#pragma unroll
	for(int i=0;i<M;i++)
		result = result + sum_copies[i];

	*output = result;
}
