//ACL Kernel

#define sr_size 180

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[sr_size];
        #pragma unroll
        for(unsigned temp = 0 ; temp <= sr_size ; temp ++ )
	{
	sum_copies[temp] = 0;
	}
	#pragma unroll 16
	for (unsigned i = 0; i < vectorSize; i++)
	{
	double cur  = (input[i]*0.5) + sum_copies[sr_size - 1];
	  #pragma unroll
          for (unsigned k = sr_size ; k > 0 ; k --)
            {
		sum_copies[k] = sum_copies[k-1];
	    }	
	 sum_copies[0] = cur ;	
	}

	#pragma unroll
	for ( unsigned j = 0 ; j < sr_size ; j++)
	{
         result = sum_copies[j] + result ;
	}

	*output = result;
}
