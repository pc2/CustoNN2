//ACL Kernel

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	for (unsigned i = 0; i < vectorSize; i++)
	{
		result = (input[i]*0.5) + result;
	}
	*output = result;
}
