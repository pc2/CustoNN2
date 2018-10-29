__kernel
void compute(   // Input data
                __global volatile int *restrict A,
                // Input size
                const int size,
                // Result
                __global int *restrict B)
{
    int i = get_global_id(0);
    int val = A[i];
    int count = 0;
    for (int j = 0; j < size; j++) {
        if (val == A[j]) {
            count++;
        }
    }
    if(count%2==1){
        B[0] = val;
    }
}