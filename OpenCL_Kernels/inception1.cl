__kernel void fuse_transpose_48_kernel0(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 192) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 192))];
  }
}

__kernel void fuse_transpose_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 192) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))];
  }
}



__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
        }
      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_1_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 96; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
        }
      }
    }
  }
}


__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 86400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 96; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 96) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_2_Conv2d_0a_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 16; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
        }
      }
    }
  }
}


__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 14400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 16; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 16) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_49_kernel0(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 192) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))];
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_3_MaxPool_0a_3x3_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 192; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? input0[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_transpose_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 192) + rc)]));
        }
      }
    }
  }
}

__kernel void InceptionV1_InceptionV1_Mixed_3b_concat(__global float* restrict T_concat, __global float* restrict input9, __global float* restrict input10, __global float* restrict input11, __global float* restrict input6, __global float* restrict input7, __global float* restrict input8, __global float* restrict input3, __global float* restrict input4, __global float* restrict input5, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((224 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) ? max(((input9[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -175616)] * input10[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) + -224)]) + input11[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) + -224)]), 0.000000e+00f) : (float)((192 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) ? max(((input6[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -150528)] * input7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) + -192)]) + input8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) + -192)]), 0.000000e+00f) : (float)((64 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) ? max(((input3[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)) + -50176)] * input4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) + -64)]) + input5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) + -64)]), 0.000000e+00f) : max(((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256))] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)]), 0.000000e+00f))));
  }
}








































