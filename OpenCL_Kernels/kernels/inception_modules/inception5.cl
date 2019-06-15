__kernel void InceptionV1_InceptionV1_Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 128; ++ff) {
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

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_9_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 32768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4d_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 128; ++ff) {
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

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_1_transpose_pad_9_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 32768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_broadcast_mul_broadcast_add_relu_transpose_broadcast_mul_broadcast_add_relu_transpose_broadcast_mul_broadcast_add_relu_concatenate_3_kernel0(__global float* restrict T_concat, __global float* restrict input9, __global float* restrict input10, __global float* restrict input11, __global float* restrict input6, __global float* restrict input7, __global float* restrict input8, __global float* restrict input3, __global float* restrict input4, __global float* restrict input5, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((448 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? max(((input9[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -87808)] * input10[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -448)]) + input11[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -448)]), 0.000000e+00f) : (float)((384 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? max(((input6[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -75264)] * input7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -384)]) + input8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -384)]), 0.000000e+00f) : (float)((128 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? max(((input3[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -25088)] * input4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -128)]) + input5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -128)]), 0.000000e+00f) : max(((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]), 0.000000e+00f))));
  }
}

/*
Missing 
1.InceptionV1/InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/Conv2D
2.InceptionV1/InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/Relu
3.InceptionV1/InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/Conv2D
4.InceptionV1/InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/Relu
5.InceptionV1/InceptionV1/Mixed_4d/Branch_3/MaxPool_0a_3x3/MaxPool
6.InceptionV1/InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/Conv2D
7.InceptionV1/InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/Relu
8.InceptionV1/InceptionV1/Mixed_4d/concat
*/
