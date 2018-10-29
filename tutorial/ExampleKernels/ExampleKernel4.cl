#define PRINTF(...) {}
//#define PRINTF(...) printf(__VA_ARGS__)

#ifndef S10
#define __fpga_reg(...) __VA_ARGS__
#endif

#define BLOCK_SIZE 64
#define PAR_A 1
#define PAR_B 16
#define PAR_READ 16

__kernel
void compute( // Input and output matrices
                 __global volatile float *restrict C,
                 __global volatile float *restrict A,
                 __global volatile float *restrict B,
                 // Dimensions of matrices.
                 const int A_height, const int A_width, const int B_width)
{
    // output blocks to compute
    for (int cx = 0; cx < A_height; cx+=BLOCK_SIZE){
        for (int cy=0; cy < B_width; cy+=BLOCK_SIZE){
            // Compute loop bounds for required input blocks
            int a_start = A_width * cx;
            int a_end   = a_start + A_width - 1;
            int b_start = cy;
            PRINTF("Working on block %d, %d, with a_start %d and b_start %d\n", cx, cy, a_start, b_start);
            float C_local[BLOCK_SIZE][BLOCK_SIZE];
            for (int a = a_start, b = b_start;
                     a <= a_end;
                     a += BLOCK_SIZE, b += (BLOCK_SIZE * B_width)){
                PRINTF("Reading a block starting at %d\n", a);
                // allocate here to allow for easy replication
                float A_local[BLOCK_SIZE][BLOCK_SIZE];
                float B_local[BLOCK_SIZE][BLOCK_SIZE];
                // fill A tile
                for (int g_idx = a, l_x = 0;
                         l_x < BLOCK_SIZE;
                         g_idx+=A_width-BLOCK_SIZE, l_x++){
                    #pragma unroll PAR_READ
                    for (int l_y = 0; l_y < BLOCK_SIZE; g_idx++, l_y++){
                        A_local[l_x][l_y] = A[g_idx];
                    }
                }
                PRINTF("Reading b block starting at %d\n", b);
                // fill transposed B tile
                for (int g_idx = b, l_x = 0;
                         l_x < BLOCK_SIZE;
                         g_idx+=B_width-BLOCK_SIZE, l_x++){
                    #pragma unroll PAR_READ
                    for (int l_y = 0; l_y < BLOCK_SIZE; g_idx++, l_y++){
                        B_local[l_y][l_x] = B[g_idx];
                    }
                }
                // compute loop
                #pragma unroll PAR_A
                for (int lcx = 0; lcx < BLOCK_SIZE; lcx++){
                    #pragma unroll PAR_B
                    for (int lcy = 0; lcy < BLOCK_SIZE; lcy++){
                        float sum = 0.0f;
                        #pragma unroll
                        for (int ab = 0; ab < BLOCK_SIZE; ab++){
                            sum += __fpga_reg(A_local[lcx][ab]) * B_local[lcy][ab];
                        }
                        C_local[lcx][lcy] = a == a_start ? sum : C_local[lcx][lcy] + sum;
                    }
                }
            }
            PRINTF("Writing c block starting at %d\n", B_width * cx + cy);
            // copy out filled C tile
            for (int g_idx = B_width * cx + cy, l_x = 0;
                     l_x < BLOCK_SIZE;
                     g_idx+=B_width-BLOCK_SIZE, l_x++){
                for (int l_y = 0; l_y < BLOCK_SIZE; g_idx++, l_y++){
                    C[g_idx] = C_local[l_x][l_y];
                }
            }
        }
    }
}