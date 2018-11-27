//ACL Kernel

#define INNER 16
#define II 16

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double cur;
        double sum_copies[II];
	
	#pragma unroll	
        for (unsigned i=0; i<II;i++)
	{
	sum_copies[i]=0.0;
	}

	
	for (unsigned i = 0; i < vectorSize; i+=INNER)
	{
	double result2 = 0.0;
        
	#pragma unroll
        for (unsigned di = 0; di < INNER; di++){
            result2 = (input[i+di]*0.5) + result2 ;
        }
	
	cur = result2 + sum_copies[II-1];
	#pragma unroll	
	for (unsigned j=II-1; j>0;j--)
	{
	sum_copies[j]= sum_copies[j-1];
	sum_copies[0]= cur;
	}
	//result += result2;
	}

	#pragma unroll
	for (unsigned i=0; i<II-1;i++)
	{

	result += sum_copies[i];
	}
	
	*output = result;
}
