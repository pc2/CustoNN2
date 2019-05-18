//ACL Kernel

#define INNER 16
#define II_CYCLES 11

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[II_CYCLES] ;
	for (unsigned i = 0; i < II_CYCLES; i++)
	{
		sum_copies[i] = 0;
	}
	for (unsigned i = 0; i < vectorSize; i+=INNER)
	{
		double result2 = 0.0;
        	#pragma unroll
        	for (unsigned di = 0; di < INNER; di++){
            		result2 = (input[i+di]*0.5) + result2;
		}
		
		double cur =  result2 + sum_copies[II_CYCLES-1];

		#pragma unroll
		for(unsigned j = II_CYCLES-1; j >0; j--)
		{
			sum_copies[j] = sum_copies[j-1];
		}
		sum_copies[0] = cur;
	
	}
	#pragma unroll
	for (unsigned i = 0; i < II_CYCLES; i++)
	{
		result += sum_copies[i];
	}
	*output = result;
}
