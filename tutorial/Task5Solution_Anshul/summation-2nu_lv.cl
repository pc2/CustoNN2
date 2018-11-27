//ACL Kernel

#define INNER 16

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	for (unsigned i = 0; i < vectorSize; i+=INNER)
	{
        #pragma unroll
		double result1 = 0.0;
        	for (unsigned di = 0; di < INNER; di++)
		{
            		result1 += (input[i+di]*0.5);
	        }
	result += result1;
	}
	*output = result;
}
