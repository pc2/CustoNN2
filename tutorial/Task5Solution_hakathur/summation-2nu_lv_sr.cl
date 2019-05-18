//ACL Kernel

#define INNER 16

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;

	double sum_copies[11];

        #pragma unroll
        for(unsigned sri = 0 ; sri < 11 ; sri ++ )
	{
	sum_copies[sri] = 0;
	}


	for (unsigned i = 0; i < vectorSize; i+=INNER)
	{
         double temp_result = 0.0 ;
	        #pragma unroll
	        for (unsigned di = 0; di < INNER; di++)
		{
	            temp_result = (input[i+di]*0.5) + temp_result;
	        }
		
	double cur  =   temp_result + sum_copies[10];
		#pragma unroll
        	for (unsigned k = 11 ; k > 0 ; k --)
            	{
		    sum_copies[k] = sum_copies[k-1];
	    	}	
	sum_copies[0] = cur ;	

	}	 

		#pragma unroll	
		for(int fi=0;fi < 11; fi++)
		{
		    result+=sum_copies[fi];
		}


        
	*output = result;
}
