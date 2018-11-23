__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	#define M 190
	double result = 0.0;
	double cur;
        double sum_copies[M];
	
	#pragma unroll	
        for (unsigned i=0; i<M-1;i++)
	{
	sum_copies[i]=0.0;
	}
        #pragma unroll 16
	for (unsigned i = 0; i < vectorSize; i++)
	{
	cur = (input[i]*0.5) + sum_copies[M-1];
	
	
	#pragma unroll	
	for (unsigned j=M-1; j>0;j--)
	{
	sum_copies[j]= sum_copies[j-1];
	sum_copies[0]= cur;
	}
		
	}
	#pragma unroll
	for (unsigned i=0; i<M-1;i++)
	{

	result += sum_copies[i];
	}
	*output = result;
}

