//ACL Kernel

#define INNER 16
//Shift Register Elements
#define SR 11

__kernel
void summation(
		  __global const double *restrict input,
		  __global double *restrict output,
		  unsigned vectorSize)

{
	double result = 0.0;

	//array to implement shift register
	double sum_copies[SR];

	//Initialize array to 0
	for(int i=0;i<SR;i++)
		sum_copies[i]=0.0;

	for (unsigned i = 0; i < vectorSize; i+=INNER)
	{
				//local variable to eliminate 'serial region'
				double tempResult= 0.0;

        #pragma unroll
        for (unsigned di = 0; di < INNER; di++){
            tempResult = (input[i+di]*0.5) + tempResult;
        }
				//add tempResult and last element of Shift register
				double cur= tempResult + sum_copies[SR-1];

				//shift register
				#pragma unroll
				for(int j=SR-1;j>0;j--)
					sum_copies[j]=sum_copies[j-1];

				//Assign the result in the first element
				sum_copies[0] = cur;

	}
	//Compute Result
	for(int i=0;i<SR;i++){
		result =result+sum_copies[i];
	}


	*output = result;
}
