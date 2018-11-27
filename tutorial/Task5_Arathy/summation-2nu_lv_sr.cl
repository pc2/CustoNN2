//ACL Kernel

#define INNER 16

#define shift_register 11


__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[shift_register];

	for(int k = 0; k<shift_register ; k++)
		sum_copies[k] = 0.0;
	

	for (unsigned i = 0; i < vectorSize; i+=INNER)
		{
			double var=0.0;

        	#pragma unroll
        	for (unsigned di = 0; di < INNER; di++)
		{
            		var = (input[i+di]*0.5) + var;
        	}
	
	//result= result + var;

	double curr = var + sum_copies[shift_register-1];
	#pragma unroll
	for (int j = shift_register-1; j>0; j--)
		sum_copies[j] = sum_copies[j-1];
	

	sum_copies[0] = curr;
}

	
	for(int i = 0; i<shift_register; i++)
	{
		result = result + sum_copies[i];

	}
	*output = result;
}


