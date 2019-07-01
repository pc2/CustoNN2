// Example program
#include <iostream>
#include <string>

int main()
{
    
float input0[64*28*28], input1[128*28*28], input2[32*28*28], input3[32*28*28],T_concat[200704];

  for (int i = 0 ; i < 64*28*28; i++){
    input0[i] = std::stof("0."+std::to_string(i));
    // input0[i] = 0;
    } 
  for (int i = 0 ; i < 128*28*28; i++){
    input1[i] = std::stof("1."+std::to_string(i));
    // input1[i] = 1;
} 
  for (int i = 0 ; i < 32*28*28; i++){
    input2[i] = std::stof("2."+std::to_string(i));
    // input2[i] = 2;
    input3[i] = std::stof("3."+std::to_string(i));
    // input3[i] = 3;
} 

for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(224 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -175616)] : (float)(192 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) ? input1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -150528)] : (float)(64 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) ? input2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -50176)] : input3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256))])));
  }

std::cout << T_concat[609] << "\n";


for (int i = 0; i < 28*28*2; i++){
      if (i == 28*28){
          std::cout << "\n";
          std::cout << "\n";
          }
      if (i % 28 == 0){
            std::cout << "\n";
          }
      std::cout << T_concat[i] << " ";
      }

}
