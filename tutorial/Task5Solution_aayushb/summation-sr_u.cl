//ACL Kernel
#define II 10
__kernel
void summation( __global const double *restrict input,__global double *restrict output,unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[II+1]; 
	for (int j=0;j<II+1; j++)
	{
		sum_copies[j] = 0;
	}
	#pragma unroll 16
 	for (unsigned i = 0; i < vectorSize; i++)
	{
		//result = (input[i]*0.5) + result;
		double curr=(input[i]*0.5) + sum_copies[II];
		#pragma unroll
		for (int k=II; k>0 ; k--)
		{
			sum_copies[k] = sum_copies[k-1];
		}
		sum_copies[0] = curr;	
	} 
	
	#pragma unroll 
	for (int i=0; i<=II ; i++)
	{
		result+=sum_copies[i];
	}



	*output = result;
}
