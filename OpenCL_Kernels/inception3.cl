__kernel void fuse_transpose_51_kernel0(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 94080; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 480))];
  }
}

__kernel void fuse_transpose_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 94080; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 480) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))];
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 192; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
        }
      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 96; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_5_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 24576; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 208; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 96; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 96) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 16; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_6_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 4096; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}


__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 48; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 16; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 16) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_52_kernel0(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 94080; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 480) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))];
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 480; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? input0[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}


__kernel void fuse_transpose_transpose_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 94080; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 480) + rc)]));
        }
      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_4b_concat(__global float* restrict T_concat, __global float* restrict input9, __global float* restrict input10, __global float* restrict input11, __global float* restrict input6, __global float* restrict input7, __global float* restrict input8, __global float* restrict input3, __global float* restrict input4, __global float* restrict input5, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((448 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? max(((input9[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -87808)] * input10[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -448)]) + input11[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -448)]), 0.000000e+00f) : (float)((400 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? max(((input6[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -78400)] * input7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -400)]) + input8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -400)]), 0.000000e+00f) : (float)((192 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? max(((input3[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512)) + -37632)] * input4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -192)]) + input5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) + -192)]), 0.000000e+00f) : max(((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]), 0.000000e+00f))));
  }
}
