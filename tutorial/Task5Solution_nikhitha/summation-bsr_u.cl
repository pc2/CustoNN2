//ACL Kernel

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[180];
	
	#pragma unroll
	for(unsigned j=0; j < 180 ; j++){
		sum_copies[j] = 0;
	}

	#pragma unroll 16
	for (unsigned i = 0; i < vectorSize; i++)
	{
		double cur = (input[i]*0.5) + sum_copies[180];
		//result = (input[i]*0.5) + result;
		#pragma unroll
		for(unsigned k = 180; k >= 1 ; k--){

			sum_copies[k] = sum_copies[k-1];

		}
		sum_copies[0] = cur;
	}
	
	#pragma unroll
	for(unsigned j=0; j < 180 ; j++){
		result += sum_copies[j];
	}
	*output = result;
}
