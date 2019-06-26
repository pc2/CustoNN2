#include <math.h>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <string>
#include <algorithm>

using namespace std;

void fusedkernel(int*  T_pad, int* input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 157323; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((458 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) < 51754)) && (2 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229) < 226)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) / 229) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52441)) + -1350)] : 0.000000e+00f);
  }
}

int main(void)
{
    int T_pad[157323];
    int input0[224*224*3];
    
    for(int i=0; i<224*224*3; i++)
    {
        input0[i] = i;
        }
        for(int i = 0 ; i < 224 ; i++)
{
   for(int j = 0 ; j < 224 ; j++)
   {
   printf("%d " ,input0[i*224+j] );
   }
   printf("\n");
} 

printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n");
fusedkernel(T_pad, input0);
for(int i = 0 ; i < 229 ; i++)
{
   for(int j = 0 ; j < 229 ; j++)
   {
   printf("%d " ,T_pad[i*229+j] );
   }
   printf("\n");
}

    }




#include <math.h>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <string>
#include <algorithm>

using namespace std;

void fusedkernel(int*  T_pad, int* input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 157323; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((458 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) < 51754)) && (2 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229) < 226)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) / 229) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52441)) + -1350)] : 0.000000e+00f);
  }
}

int main(void)
{
    int T_pad[157323];
    int input0[224*224*3];
    
    for(int i=0; i<224*224*3; i++)
    {
        input0[i] = i;
        }
        for(int i = 0 ; i < 224 ; i++)
{
   for(int j = 0 ; j < 224 ; j++)
   {
   printf("%d " ,input0[i*224+j] );
   }
   printf("\n");
} 

printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n");
fusedkernel(T_pad, input0);
for(int i = 0 ; i < 229 ; i++)
{
   for(int j = 0 ; j < 229 ; j++)
   {
   printf("%d " ,T_pad[i*229+j] );
   }
   printf("\n");
}

    }





