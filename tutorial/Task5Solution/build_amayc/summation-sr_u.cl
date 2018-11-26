//ACL Kernel

__kernel void summation(__global const double *restrict input, __global double *restrict output, unsigned vectorSize)
{
	double result = 0.0;
	double sum_copies[11] ;
	for (unsigned i = 0; i < 11; i++)
	{
		sum_copies[i] = 0;
	}
	#pragma unroll 16
	for (unsigned i = 0; i < vectorSize; i++)
	{
		double cur =  (input[i]*0.5) + sum_copies[10];
		 
		//result = (input[i]*0.5) + result;

		#pragma unroll
		for(unsigned j = 10; j >0; j--)
		{
			sum_copies[j] = sum_copies[j-1];

		}
		sum_copies[0] = cur;
	}
	#pragma unroll
	for (unsigned i = 0; i < 11; i++)
	{
		result += sum_copies[i];
	}
	*output = result;
}
