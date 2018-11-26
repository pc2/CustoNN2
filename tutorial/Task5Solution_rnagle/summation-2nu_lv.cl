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
	double result_2=0.0;
        #pragma unroll
        for (unsigned di = 0; di < INNER; di++){
            result_2 = (input[i+di]*0.5) + result_2;
        }
	result+=result_2;
	}
	
	*output = result;
}
