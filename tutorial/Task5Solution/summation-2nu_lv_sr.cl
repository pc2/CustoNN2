//ACL Kernel

#define INNER 16

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;
	double shift_register[180];
	#pragma unroll
	for(unsigned j=0; j < 180 ; j++){
		shift_register[j] = 0;
	}

	for (unsigned i = 0; i < vectorSize; i+=INNER){
        	#pragma unroll
		double localtemp = 0.0;
        	for (unsigned di = 0; di < INNER; di++){
			localtemp = (input[i+di]*0.5) + localtemp;	
        	}
		double cur = localtemp + shift_register[179];
		for(unsigned k = 180; k >= 1 ; k--){
			shift_register[k] = shift_register[k-1];
		}
		shift_register[0]=cur;
	}

	#pragma unroll
	for(unsigned j=0; j < 180 ; j++){
		result += shift_register[j];
	}
	*output = result;
}
