#define MASK_SIZE 3

__kernel void compute(
        __global float *inputImage,
        __constant float *mask,
        __global float *outputImage
    ) {
    // load mask values
    __local float local_mask[MASK_SIZE][MASK_SIZE];
    const int2 pos = {get_global_id(0), get_global_id(1)};
    if(pos.x < MASK_SIZE && pos.y < MASK_SIZE){
        local_mask[pos.y][pos.x] = mask[pos.x + pos.y*MASK_SIZE];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // compute
    float sum = 0.0f;
    for(int a = 0; a < MASK_SIZE; a++) {
        for(int b = 0; b < MASK_SIZE; b++) {
            const int2 otherPos = pos + (int2){a, b} - (int2){MASK_SIZE/2, MASK_SIZE/2};
            float inputVal = ( otherPos.x >= 0 && otherPos.x < get_global_size(0) && otherPos.y >= 0 && otherPos.y < get_global_size(1) ) ?
                             inputImage[pos.x+pos.y*get_global_size(0)] :
                             0.0;
            sum += local_mask[a][b] * inputVal;
        }
    }

    outputImage[pos.x+pos.y*get_global_size(0)] = sum;
}