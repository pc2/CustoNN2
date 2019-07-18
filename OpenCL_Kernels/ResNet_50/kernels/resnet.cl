__kernel void fuse_pad_transpose_kernel0(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 158700; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((690 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52900) < 52210)) && (3 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 230))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 230) < 227)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52900) / 230) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 230)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52900)) + -2025)] : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        compute[((((ff * 112) + yy) * 112) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 3; ++rc) {
          for (int ry = 0; ry < 7; ++ry) {
            for (int rx = 0; rx < 7; ++rx) {
              compute[((((ff * 112) + yy) * 112) + xx)] = (compute[((((ff * 112) + yy) * 112) + xx)] + (input0[((((((((rc * 115) + yy) * 2) + ry) * 115) + xx) * 2) + rx)] * input1[((((((ff * 3) + rc) * 7) + ry) * 7) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_add_transpose_kernel0(__global float* restrict T_transpose, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 12544)]);
  }
}

__kernel void fuse_max_pool2d_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 64; ++ax1) {
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = max(tensor[((((ax1 * 56) + ax2) * 56) + ax3)], (float)((((ax2 * 2) < (112 - rv)) && ((ax3 * 2) < (112 - rv1))) ? input0[((((((((ax1 * 56) + ax2) * 2) + rv) * 56) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_kernel0(__global float* restrict T_relu, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_relu[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 64) * 3136) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 64))] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 64)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 64)]), 0.000000e+00f);
  }
}

__kernel void fuse_transpose_pad_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3136) * 64) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136))];
  }
}

__kernel void fuse_conv2d_1_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        compute[((((ff * 56) + yy) * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 64) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_conv2d_2_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        compute[((((ff * 56) + yy) * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 64) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_3_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        compute[((((ff * 56) + yy) * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (input0[((((((rc * 58) + yy) + ry) * 58) + xx) + rx)] * input1[((((((ff * 64) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]), 0.000000e+00f);
  }
}

__kernel void fuse_transpose_broadcast_add_transpose_broadcast_add_broadcast_add_kernel0(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = ((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 3136) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256))] + input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)]) + (input2[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 3136) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256))] + input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)]));
  }
}

__kernel void fuse_broadcast_mul_broadcast_add_relu_transpose_pad_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3136) * 256) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136))] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_4_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        compute[((((ff * 56) + yy) * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 256) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_add_broadcast_add_kernel0(__global float* restrict T_add, __global float* restrict input2, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (input2[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + (input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 3136) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256))] + input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)]));
  }
}

__kernel void fuse_transpose_21_kernel0(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3136) * 256) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136))];
  }
}

__kernel void fuse_max_pool2d_1_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], input0[(((((ax1 * 28) + ax2) * 56) + ax3) * 2)]);
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_pad_transpose_kernel0(__global float* restrict T_transpose, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_5_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((((rc * 29) + yy) * 2) + ry) * 29) + xx) * 2) + rx)] * input1[((((((ff * 64) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_6_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 64) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_transpose_broadcast_add_broadcast_add_broadcast_mul_broadcast_add_relu_kernel0(__global float* restrict T_relu, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3, __global float* restrict input4) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_relu[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max((((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256))] + (input1[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256))] + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)])) * input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)]) + input4[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)]), 0.000000e+00f);
  }
}

__kernel void fuse_transpose_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 256) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))];
  }
}

__kernel void fuse_conv2d_7_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_conv2d_8_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_9_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_10_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 128) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_add_transpose_broadcast_add_broadcast_add_1_kernel0(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = ((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))] + input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]) + (input2[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))] + input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]));
  }
}

__kernel void fuse_broadcast_mul_broadcast_add_relu_transpose_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 512) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_11_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 512) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_add_broadcast_add_1_kernel0(__global float* restrict T_add, __global float* restrict input2, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (input2[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + (input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 784) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))] + input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]));
  }
}

__kernel void fuse_transpose_22_kernel0(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 784) * 512) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784))];
  }
}

__kernel void fuse_max_pool2d_2_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], input0[(((((ax1 * 14) + ax2) * 28) + ax3) * 2)]);
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_pad_transpose_1_kernel0(__global float* restrict T_transpose, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_12_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((((rc * 15) + yy) * 2) + ry) * 15) + xx) * 2) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_5_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 25088; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_13_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 128) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_transpose_broadcast_add_broadcast_add_broadcast_mul_broadcast_add_relu_1_kernel0(__global float* restrict T_relu, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3, __global float* restrict input4) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_relu[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max((((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))] + (input1[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512))] + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)])) * input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]) + input4[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]), 0.000000e+00f);
  }
}

__kernel void fuse_transpose_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 512) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))];
  }
}

__kernel void fuse_conv2d_14_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 1024; ++ff) {
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

__kernel void fuse_conv2d_15_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 256; ++ff) {
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

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_6_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 65536; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_16_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * input1[((((((ff * 256) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_7_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_17_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 1024; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 256) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_add_transpose_broadcast_add_broadcast_add_2_kernel0(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = ((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1024))] + input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)]) + (input2[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1024))] + input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)]));
  }
}

__kernel void fuse_broadcast_mul_broadcast_add_relu_transpose_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 1024) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_18_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 1024; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 1024) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_add_broadcast_add_2_kernel0(__global float* restrict T_add, __global float* restrict input2, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (input2[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + (input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024) * 196) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1024))] + input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)]));
  }
}

__kernel void fuse_transpose_23_kernel0(__global float* restrict T_transpose, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 196) * 1024) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196))];
  }
}

__kernel void fuse_max_pool2d_3_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], input0[(((((ax1 * 7) + ax2) * 14) + ax3) * 2)]);
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_pad_transpose_2_kernel0(__global float* restrict T_transpose, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 65536; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -15)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_19_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((((((rc * 8) + yy) * 2) + ry) * 8) + xx) * 2) + rx)] * input1[((((((ff * 256) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_8_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 12544; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_20_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 1024; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 256) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_transpose_broadcast_add_broadcast_add_broadcast_mul_broadcast_add_relu_2_kernel0(__global float* restrict T_relu, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3, __global float* restrict input4) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_relu[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max((((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1024))] + (input1[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1024))] + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)])) * input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)]) + input4[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)]), 0.000000e+00f);
  }
}

__kernel void fuse_transpose_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 49) * 1024) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49))];
  }
}

__kernel void fuse_conv2d_21_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 2048; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 1024; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 1024) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_conv2d_22_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 1024; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 1024) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_9_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 41472; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? max(((input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + -8)] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_23_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * input1[((((((ff * 512) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_mul_broadcast_add_relu_transpose_pad_10_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 25088; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_24_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 2048; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 512) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_add_transpose_broadcast_add_broadcast_add_3_kernel0(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = ((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 2048))] + input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048)]) + (input2[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 2048))] + input3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048)]));
  }
}

__kernel void fuse_broadcast_mul_broadcast_add_relu_transpose_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 49) * 2048) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49))] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_25_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 2048; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (input0[((((rc * 7) + yy) * 7) + xx)] * input1[((ff * 2048) + rc)]));
        }
      }
    }
  }
}

__kernel void fuse_transpose_broadcast_add_broadcast_add_3_kernel0(__global float* restrict T_add, __global float* restrict input2, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (input2[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + (input0[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048) * 49) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 2048))] + input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048)]));
  }
}

__kernel void fuse_transpose_broadcast_add_broadcast_add_broadcast_mul_broadcast_add_relu_mean_kernel0(__global float* restrict T_relu_red, __global float* restrict input2, __global float* restrict input0, __global float* restrict input1, __global float* restrict input3, __global float* restrict input4, __global float* restrict T_divide) {
  for (int ax3 = 0; ax3 < 2048; ++ax3) {
    T_relu_red[ax3] = 0.000000e+00f;
    for (int k1 = 0; k1 < 7; ++k1) {
      for (int k2 = 0; k2 < 7; ++k2) {
        T_relu_red[ax3] = (max((((input2[((((k1 * 7) + k2) * 2048) + ax3)] + (input0[((((ax3 * 7) + k1) * 7) + k2)] + input1[ax3])) * input3[ax3]) + input4[ax3]), 0.000000e+00f) + T_relu_red[ax3]);
      }
    }
  }
  for (int ax31 = 0; ax31 < 2048; ++ax31) {
    T_divide[ax31] = (T_relu_red[ax31] * 2.040816e-02f);
  }
}

__kernel void fuse_transpose_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 2048; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_26_kernel0(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1) {
  for (int ff = 0; ff < 1001; ++ff) {
    compute[ff] = 0.000000e+00f;
    for (int rc = 0; rc < 2048; ++rc) {
      compute[ff] = (compute[ff] + (input0[rc] * input1[((ff * 2048) + rc)]));
    }
  }
}

__kernel void fuse_transpose_broadcast_add_squeeze_reshape_flatten_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    tensor[ax0_ax1_fused_inner] = (input0[ax0_ax1_fused_inner] + input1[ax0_ax1_fused_inner]);
  }
}

__kernel void fuse_softmax_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict tensor1, __global float* restrict tensor2) {
  for (int ax1 = 0; ax1 < 1001; ++ax1) {
    tensor[0] = -3.402823e+38f;
    for (int k1 = 0; k1 < 1001; ++k1) {
      tensor[0] = max(tensor[0], input0[k1]);
    }
    tensor1[0] = 0.000000e+00f;
    for (int k2 = 0; k2 < 1001; ++k2) {
      tensor1[0] = (tensor1[0] + exp((input0[k2] - tensor[0])));
    }
    tensor2[ax1] = (exp((input0[ax1] - tensor[0])) / tensor1[0]);
  }
}

__kernel void fuse_reshape_kernel0(__global float* restrict T_reshape, __global float* restrict input0) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    T_reshape[ax0_ax1_fused_inner] = input0[ax0_ax1_fused_inner];
  }
}


