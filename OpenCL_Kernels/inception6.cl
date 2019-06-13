__kernel void InceptionV1_InceptionV1_Mixed_4e_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 144; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_10_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 36864; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 288; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 144; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 144) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4e_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 512) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_11_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}


__kernel void InceptionV1_InceptionV1_Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 32; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}


/*
Missing 
1.InceptionV1/InceptionV1/Mixed_4e/Branch_3/MaxPool_0a_3x3/MaxPool
2.InceptionV1/InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/Conv2D
3.InceptionV1/InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/Relu
4.InceptionV1/InceptionV1/Mixed_4e/concat
*/

