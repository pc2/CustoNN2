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
	double result2 = 0.0;
        #pragma unroll
        for (unsigned di = 0; di < INNER; di++){
            result2 = (input[i+di]*0.5) + result2;
        }
	result += result2;
	
	}
	*output = result;
}
