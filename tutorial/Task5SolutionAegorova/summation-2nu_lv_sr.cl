//ACL Kernel

#define INNER 16
#define II 11

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double sum_copies[II];

	for (unsigned i = 0; i < II; i++) {
		sum_copies[i] = 0;
	}


	for (unsigned i = 0; i < vectorSize; i+=INNER)
	{

		double result2= 0.0;
		
		#pragma unroll
		for (unsigned di = 0; di < INNER; di++){
		    //result  += (input[i+di]*0.5) + result;
			result2 += (input[i+di]*0.5);
		}

		//result += result2;
		double cur = result2 + sum_copies[II-1];
		for (unsigned i = II-1; i > 0; i--) {
			sum_copies[i] = sum_copies[i-1];
		}
		sum_copies[0] = cur; 
		
	}
	#pragma unroll
	for (unsigned i = 0; i < II; i++) {
		result = result + sum_copies[i];
	}
	*output = result;
}
