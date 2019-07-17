nclude <math.h>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <string>
#include <algorithm>

using namespace std;

void Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D(int*  compute,  int*  input0,  int*  input1 ,  int*  input2)
{
 for (int ff = 0; ff < 2; ++ff)

 {
   for (int yy = 0; yy < 28; ++yy)
   {
     for (int xx = 0; xx < 28; ++xx)
     {
       compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];

       for (int rc = 0; rc < 4; ++rc)
       {
         compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 2) + rc)]));
       }
       compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)]>0)? compute[((((ff * 28) + yy) * 28) + xx)]:0;
     }
   }
 }


    if(1)
    {
       printf("Supra conv 3c \n");
       for (int i = 0 ; i < 28 ; i++)
        {
            for(int j = 0 ; j < 28 ; j++)
            {
                printf("%d ",compute[i*28 + j]);            
            }    
            printf("\n");
        }
    }


}



int main(void)
{


int  compute[812*2] ;
int  input0[2380*2] ;
int  input1[2*2] = {1,1,1,1} ;

int  input2[1*2] = {1,1} ;



for(int i = 0 ; i <2380*2 ; i++)
{
input0[i] = 1;
}








printf("################################################################################################################ \n");
Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D(compute,input0,input1 ,input2);

if(0)
{
   for(int i = 0 ; i < 28 ; i++)
   {
    for(int j = 0 ; j < 28 ; j++)
   {
   printf("%d " ,compute[i*28+j]);
   }
   printf("\n");
   }
}


}
