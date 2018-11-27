//ACL Kernel

#define INNER 16

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double result2[10];
	#pragma unroll
	for ( int j=0; j<10;j++)
	{
		result2[j]=0;
	}	
	for (unsigned i = 0; i < vectorSize; i+=INNER)
	{
        #pragma unroll
		double result1 = 0.0;
        	for (unsigned di = 0; di < INNER; di++)
		{
            		result1 += (input[i+di]*0.5);
	        }
		double r = result1 + result2[9];
		for (int k=10;k>0;k--)
		{
			result2[k]=result2[k-1];
		}
		result2[0]=r;
	}
	#pragma unroll
	for ( int l=0;l<10;l++)
	{
		result += result2[l];
	}
	*output = result;
}
