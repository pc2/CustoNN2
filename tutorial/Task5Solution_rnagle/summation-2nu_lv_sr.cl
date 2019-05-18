//ACL Kernel

#define INNER 16
#define II 10

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[II+1];
	for(int i=0;i<=II;i++)
	{
		sum_copies[i]=0;
	}
	

	for (unsigned i = 0; i < vectorSize; i+=INNER)
	{
	double result_2=0.0;

        #pragma unroll
        for (unsigned di = 0; di < INNER; di++){
            result_2 = (input[i+di]*0.5) + result_2;
        }

	//result+=result_2;
	double curr=result_2+sum_copies[II];
	#pragma unroll
	for(int j=II;j>0;j--)
	{
		sum_copies[j]=sum_copies[j-1];
	}
	sum_copies[0]=curr;
	}
	
	#pragma unroll	
	for(int j=0;j<=II;j++)
	{
		result+=sum_copies[j];
	}

	*output = result;
}
