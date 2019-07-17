__kernel void fuse_pad_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 158700; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((2070 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 156630)) && (9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 690))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 690) < 681)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 690) * 672) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 690)) + -2025)] : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_broadcast_add_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 112; ++ax1) {
    for (int xx = 0; xx < 112; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 7; ++ry) {
          for (int rx = 0; rx < 7; ++rx) {
            for (int rc = 0; rc < 3; ++rc) {
              Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (input0[((((((((ax1 * 2) + ry) * 115) + xx) * 2) + rx) * 3) + rc)] * input1[((((((ry * 7) + rx) * 3) + rc) * 64) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_add[((((ax1 * 112) + ax2) * 64) + ax3)] = (Conv2dOutput[((ax2 * 64) + ax3)] + input2[ax3]);
      }
    }
  }
}

__kernel void fuse_max_pool2d_broadcast_mul_broadcast_add_relu_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict T_relu, __global float* restrict input1, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        tensor[((ax2 * 64) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((ax2 * 64) + ax3)] = max(tensor[((ax2 * 64) + ax3)], (float)((((ax1 * 2) < (112 - rv)) && ((ax2 * 2) < (112 - rv1))) ? input0[((((((((ax1 * 2) + rv) * 56) + ax2) * 2) + rv1) * 64) + ax3)] : -3.402823e+38f));
          }
        }
      }
    }
    for (int ax21 = 0; ax21 < 56; ++ax21) {
      for (int ax31 = 0; ax31 < 64; ++ax31) {
        T_relu[((((ax1 * 56) + ax21) * 64) + ax31)] = max(((tensor[((ax21 * 64) + ax31)] * input1[ax31]) + input2[ax31]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int xx = 0; xx < 56; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (input0[((((ax1 * 56) + xx) * 64) + rc)] * input1[((rc * 64) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 56) + ax2) * 64) + ax3)] = max(((Conv2dOutput[((ax2 * 64) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((3712 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 211584)) && (64 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3712))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3712) < 3648)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3712) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3712)) + -3648)] : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_1_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int xx = 0; xx < 56; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 64; ++rc) {
              Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (input0[((((((ax1 + ry) * 58) + xx) + rx) * 64) + rc)] * input1[((((((ry * 3) + rx) * 64) + rc) * 64) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 56) + ax2) * 64) + ax3)] = max(((Conv2dOutput[((ax2 * 64) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_1_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int xx = 0; xx < 56; ++xx) {
      for (int ff = 0; ff < 256; ++ff) {
        Conv2dOutput[((xx * 256) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          Conv2dOutput[((xx * 256) + ff)] = (Conv2dOutput[((xx * 256) + ff)] + (input0[((((ax1 * 56) + xx) * 64) + rc)] * input1[((rc * 256) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        T_add[((((ax1 * 56) + ax2) * 256) + ax3)] = (Conv2dOutput[((ax2 * 256) + ax3)] + input2[ax3]);
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_broadcast_add_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int xx = 0; xx < 56; ++xx) {
      for (int ff = 0; ff < 256; ++ff) {
        Conv2dOutput[((xx * 256) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          Conv2dOutput[((xx * 256) + ff)] = (Conv2dOutput[((xx * 256) + ff)] + (input0[((((ax1 * 56) + xx) * 64) + rc)] * input1[((rc * 256) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        T_add[((((ax1 * 56) + ax2) * 256) + ax3)] = ((Conv2dOutput[((ax2 * 256) + ax3)] + input2[ax3]) + input3[((((ax1 * 56) + ax2) * 256) + ax3)]);
      }
    }
  }
}

__kernel void fuse_broadcast_mul_broadcast_add_relu_pad_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_2_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int xx = 0; xx < 56; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (input0[((((ax1 * 56) + xx) * 256) + rc)] * input1[((rc * 64) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 56) + ax2) * 64) + ax3)] = max(((Conv2dOutput[((ax2 * 64) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_broadcast_add_1_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input3, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int xx = 0; xx < 56; ++xx) {
      for (int ff = 0; ff < 256; ++ff) {
        Conv2dOutput[((xx * 256) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          Conv2dOutput[((xx * 256) + ff)] = (Conv2dOutput[((xx * 256) + ff)] + (input0[((((ax1 * 56) + xx) * 64) + rc)] * input1[((rc * 256) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        T_add[((((ax1 * 56) + ax2) * 256) + ax3)] = (input3[((((ax1 * 56) + ax2) * 256) + ax3)] + (Conv2dOutput[((ax2 * 256) + ax3)] + input2[ax3]));
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_3_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 64; ++rc) {
              Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (input0[((((((((ax1 * 2) + ry) * 29) + xx) * 2) + rx) * 64) + rc)] * input1[((((((ry * 3) + rx) * 64) + rc) * 64) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 64) + ax3)] = max(((Conv2dOutput[((ax2 * 64) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_add_2_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 256; ++ff) {
        Conv2dOutput[((xx * 256) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          Conv2dOutput[((xx * 256) + ff)] = (Conv2dOutput[((xx * 256) + ff)] + (input0[((((ax1 * 28) + xx) * 64) + rc)] * input1[((rc * 256) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        T_add[((((ax1 * 28) + ax2) * 256) + ax3)] = (Conv2dOutput[((ax2 * 256) + ax3)] + input2[ax3]);
      }
    }
  }
}

__kernel void fuse_max_pool2d_broadcast_add_broadcast_mul_broadcast_add_relu_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict T_relu, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        tensor[((ax2 * 256) + ax3)] = -3.402823e+38f;
        tensor[((ax2 * 256) + ax3)] = max(tensor[((ax2 * 256) + ax3)], input0[((((ax1 * 56) + ax2) * 512) + ax3)]);
      }
    }
    for (int ax21 = 0; ax21 < 28; ++ax21) {
      for (int ax31 = 0; ax31 < 256; ++ax31) {
        T_relu[((((ax1 * 28) + ax21) * 256) + ax31)] = max((((tensor[((ax21 * 256) + ax31)] + input1[((((ax1 * 28) + ax21) * 256) + ax31)]) * input2[ax31]) + input3[ax31]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_4_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (input0[((((ax1 * 28) + xx) * 256) + rc)] * input1[((rc * 128) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 128) + ax3)] = max(((Conv2dOutput[((ax2 * 128) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_5_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((3840 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 111360)) && (128 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3840))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3840) < 3712)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3840) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3840)) + -3712)] : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_5_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 128; ++rc) {
              Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (input0[((((((ax1 + ry) * 30) + xx) + rx) * 128) + rc)] * input1[((((((ry * 3) + rx) * 128) + rc) * 128) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 128) + ax3)] = max(((Conv2dOutput[((ax2 * 128) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_6_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_add_3_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 512; ++ff) {
        Conv2dOutput[((xx * 512) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          Conv2dOutput[((xx * 512) + ff)] = (Conv2dOutput[((xx * 512) + ff)] + (input0[((((ax1 * 28) + xx) * 128) + rc)] * input1[((rc * 512) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 512; ++ax3) {
        T_add[((((ax1 * 28) + ax2) * 512) + ax3)] = (Conv2dOutput[((ax2 * 512) + ax3)] + input2[ax3]);
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_broadcast_add_2_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 512; ++ff) {
        Conv2dOutput[((xx * 512) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          Conv2dOutput[((xx * 512) + ff)] = (Conv2dOutput[((xx * 512) + ff)] + (input0[((((ax1 * 28) + xx) * 256) + rc)] * input1[((rc * 512) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 512; ++ax3) {
        T_add[((((ax1 * 28) + ax2) * 512) + ax3)] = ((Conv2dOutput[((ax2 * 512) + ax3)] + input2[ax3]) + input3[((((ax1 * 28) + ax2) * 512) + ax3)]);
      }
    }
  }
}

__kernel void fuse_broadcast_mul_broadcast_add_relu_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_6_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (input0[((((ax1 * 28) + xx) * 512) + rc)] * input1[((rc * 128) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 128) + ax3)] = max(((Conv2dOutput[((ax2 * 128) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_broadcast_add_3_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input3, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 512; ++ff) {
        Conv2dOutput[((xx * 512) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          Conv2dOutput[((xx * 512) + ff)] = (Conv2dOutput[((xx * 512) + ff)] + (input0[((((ax1 * 28) + xx) * 128) + rc)] * input1[((rc * 512) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 512; ++ax3) {
        T_add[((((ax1 * 28) + ax2) * 512) + ax3)] = (input3[((((ax1 * 28) + ax2) * 512) + ax3)] + (Conv2dOutput[((ax2 * 512) + ax3)] + input2[ax3]));
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_7_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 128; ++rc) {
              Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (input0[((((((((ax1 * 2) + ry) * 15) + xx) * 2) + rx) * 128) + rc)] * input1[((((((ry * 3) + rx) * 128) + rc) * 128) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 128) + ax3)] = max(((Conv2dOutput[((ax2 * 128) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_7_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 25088; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_add_4_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 512; ++ff) {
        Conv2dOutput[((xx * 512) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          Conv2dOutput[((xx * 512) + ff)] = (Conv2dOutput[((xx * 512) + ff)] + (input0[((((ax1 * 14) + xx) * 128) + rc)] * input1[((rc * 512) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 512; ++ax3) {
        T_add[((((ax1 * 14) + ax2) * 512) + ax3)] = (Conv2dOutput[((ax2 * 512) + ax3)] + input2[ax3]);
      }
    }
  }
}

__kernel void fuse_max_pool2d_broadcast_add_broadcast_mul_broadcast_add_relu_1_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict T_relu, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 512; ++ax3) {
        tensor[((ax2 * 512) + ax3)] = -3.402823e+38f;
        tensor[((ax2 * 512) + ax3)] = max(tensor[((ax2 * 512) + ax3)], input0[((((ax1 * 28) + ax2) * 1024) + ax3)]);
      }
    }
    for (int ax21 = 0; ax21 < 14; ++ax21) {
      for (int ax31 = 0; ax31 < 512; ++ax31) {
        T_relu[((((ax1 * 14) + ax21) * 512) + ax31)] = max((((tensor[((ax21 * 512) + ax31)] + input1[((((ax1 * 14) + ax21) * 512) + ax31)]) * input2[ax31]) + input3[ax31]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_8_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_8_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 256; ++ff) {
        Conv2dOutput[((xx * 256) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 256) + ff)] = (Conv2dOutput[((xx * 256) + ff)] + (input0[((((ax1 * 14) + xx) * 512) + rc)] * input1[((rc * 256) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 256) + ax3)] = max(((Conv2dOutput[((ax2 * 256) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_9_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 65536; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((4096 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 61440)) && (256 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 4096))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 4096) < 3840)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 4096) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 4096)) + -3840)] : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_9_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 256; ++ff) {
        Conv2dOutput[((xx * 256) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 256; ++rc) {
              Conv2dOutput[((xx * 256) + ff)] = (Conv2dOutput[((xx * 256) + ff)] + (input0[((((((ax1 + ry) * 16) + xx) + rx) * 256) + rc)] * input1[((((((ry * 3) + rx) * 256) + rc) * 256) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 256) + ax3)] = max(((Conv2dOutput[((ax2 * 256) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_10_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_add_5_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 1024; ++ff) {
        Conv2dOutput[((xx * 1024) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          Conv2dOutput[((xx * 1024) + ff)] = (Conv2dOutput[((xx * 1024) + ff)] + (input0[((((ax1 * 14) + xx) * 256) + rc)] * input1[((rc * 1024) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 1024; ++ax3) {
        T_add[((((ax1 * 14) + ax2) * 1024) + ax3)] = (Conv2dOutput[((ax2 * 1024) + ax3)] + input2[ax3]);
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_broadcast_add_4_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 1024; ++ff) {
        Conv2dOutput[((xx * 1024) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 1024) + ff)] = (Conv2dOutput[((xx * 1024) + ff)] + (input0[((((ax1 * 14) + xx) * 512) + rc)] * input1[((rc * 1024) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 1024; ++ax3) {
        T_add[((((ax1 * 14) + ax2) * 1024) + ax3)] = ((Conv2dOutput[((ax2 * 1024) + ax3)] + input2[ax3]) + input3[((((ax1 * 14) + ax2) * 1024) + ax3)]);
      }
    }
  }
}

__kernel void fuse_broadcast_mul_broadcast_add_relu_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_10_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 256; ++ff) {
        Conv2dOutput[((xx * 256) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 1024; ++rc) {
          Conv2dOutput[((xx * 256) + ff)] = (Conv2dOutput[((xx * 256) + ff)] + (input0[((((ax1 * 14) + xx) * 1024) + rc)] * input1[((rc * 256) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 256) + ax3)] = max(((Conv2dOutput[((ax2 * 256) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_broadcast_add_5_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input3, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 1024; ++ff) {
        Conv2dOutput[((xx * 1024) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          Conv2dOutput[((xx * 1024) + ff)] = (Conv2dOutput[((xx * 1024) + ff)] + (input0[((((ax1 * 14) + xx) * 256) + rc)] * input1[((rc * 1024) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 1024; ++ax3) {
        T_add[((((ax1 * 14) + ax2) * 1024) + ax3)] = (input3[((((ax1 * 14) + ax2) * 1024) + ax3)] + (Conv2dOutput[((ax2 * 1024) + ax3)] + input2[ax3]));
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_11_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 256; ++ff) {
        Conv2dOutput[((xx * 256) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 256; ++rc) {
              Conv2dOutput[((xx * 256) + ff)] = (Conv2dOutput[((xx * 256) + ff)] + (input0[((((((((ax1 * 2) + ry) * 8) + xx) * 2) + rx) * 256) + rc)] * input1[((((((ry * 3) + rx) * 256) + rc) * 256) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 256) + ax3)] = max(((Conv2dOutput[((ax2 * 256) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_11_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 12544; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_add_6_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 1024; ++ff) {
        Conv2dOutput[((xx * 1024) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          Conv2dOutput[((xx * 1024) + ff)] = (Conv2dOutput[((xx * 1024) + ff)] + (input0[((((ax1 * 7) + xx) * 256) + rc)] * input1[((rc * 1024) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 1024; ++ax3) {
        T_add[((((ax1 * 7) + ax2) * 1024) + ax3)] = (Conv2dOutput[((ax2 * 1024) + ax3)] + input2[ax3]);
      }
    }
  }
}

__kernel void fuse_max_pool2d_broadcast_add_broadcast_mul_broadcast_add_relu_2_kernel0(__global float* restrict tensor, __global float* restrict input0, __global float* restrict T_relu, __global float* restrict input1, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 1024; ++ax3) {
        tensor[((ax2 * 1024) + ax3)] = -3.402823e+38f;
        tensor[((ax2 * 1024) + ax3)] = max(tensor[((ax2 * 1024) + ax3)], input0[((((ax1 * 14) + ax2) * 2048) + ax3)]);
      }
    }
    for (int ax21 = 0; ax21 < 7; ++ax21) {
      for (int ax31 = 0; ax31 < 1024; ++ax31) {
        T_relu[((((ax1 * 7) + ax21) * 1024) + ax31)] = max((((tensor[((ax21 * 1024) + ax31)] + input1[((((ax1 * 7) + ax21) * 1024) + ax31)]) * input2[ax31]) + input3[ax31]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_12_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_12_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 512; ++ff) {
        Conv2dOutput[((xx * 512) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 1024; ++rc) {
          Conv2dOutput[((xx * 512) + ff)] = (Conv2dOutput[((xx * 512) + ff)] + (input0[((((ax1 * 7) + xx) * 1024) + rc)] * input1[((rc * 512) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 512; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 512) + ax3)] = max(((Conv2dOutput[((ax2 * 512) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_13_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 41472; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((4608 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 36864)) && (512 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 4608))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 4608) < 4096)) ? input0[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 4608) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 4608)) + -4096)] : 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_13_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 512; ++ff) {
        Conv2dOutput[((xx * 512) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 512; ++rc) {
              Conv2dOutput[((xx * 512) + ff)] = (Conv2dOutput[((xx * 512) + ff)] + (input0[((((((ax1 + ry) * 9) + xx) + rx) * 512) + rc)] * input1[((((((ry * 3) + rx) * 512) + rc) * 512) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 512; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 512) + ax3)] = max(((Conv2dOutput[((ax2 * 512) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_pad_14_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 25088; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_add_7_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 2048; ++ff) {
        Conv2dOutput[((xx * 2048) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 2048) + ff)] = (Conv2dOutput[((xx * 2048) + ff)] + (input0[((((ax1 * 7) + xx) * 512) + rc)] * input1[((rc * 2048) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 2048; ++ax3) {
        T_add[((((ax1 * 7) + ax2) * 2048) + ax3)] = (Conv2dOutput[((ax2 * 2048) + ax3)] + input2[ax3]);
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_broadcast_add_6_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 2048; ++ff) {
        Conv2dOutput[((xx * 2048) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 1024; ++rc) {
          Conv2dOutput[((xx * 2048) + ff)] = (Conv2dOutput[((xx * 2048) + ff)] + (input0[((((ax1 * 7) + xx) * 1024) + rc)] * input1[((rc * 2048) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 2048; ++ax3) {
        T_add[((((ax1 * 7) + ax2) * 2048) + ax3)] = ((Conv2dOutput[((ax2 * 2048) + ax3)] + input2[ax3]) + input3[((((ax1 * 7) + ax2) * 2048) + ax3)]);
      }
    }
  }
}

__kernel void fuse_broadcast_mul_broadcast_add_relu_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048)]), 0.000000e+00f);
  }
}

__kernel void fuse_conv2d_broadcast_mul_broadcast_add_relu_14_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input2, __global float* restrict input3) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 512; ++ff) {
        Conv2dOutput[((xx * 512) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 2048; ++rc) {
          Conv2dOutput[((xx * 512) + ff)] = (Conv2dOutput[((xx * 512) + ff)] + (input0[((((ax1 * 7) + xx) * 2048) + rc)] * input1[((rc * 512) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 512; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 512) + ax3)] = max(((Conv2dOutput[((ax2 * 512) + ax3)] * input2[ax3]) + input3[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_broadcast_add_7_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input3, __global float* restrict input2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 2048; ++ff) {
        Conv2dOutput[((xx * 2048) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 2048) + ff)] = (Conv2dOutput[((xx * 2048) + ff)] + (input0[((((ax1 * 7) + xx) * 512) + rc)] * input1[((rc * 2048) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 2048; ++ax3) {
        T_add[((((ax1 * 7) + ax2) * 2048) + ax3)] = (input3[((((ax1 * 7) + ax2) * 2048) + ax3)] + (Conv2dOutput[((ax2 * 2048) + ax3)] + input2[ax3]));
      }
    }
  }
}

__kernel void fuse_conv2d_broadcast_add_broadcast_add_broadcast_mul_broadcast_add_relu_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_relu, __global float* restrict input3, __global float* restrict input2, __global float* restrict input4, __global float* restrict input5) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 2048; ++ff) {
        Conv2dOutput[((xx * 2048) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 2048) + ff)] = (Conv2dOutput[((xx * 2048) + ff)] + (input0[((((ax1 * 7) + xx) * 512) + rc)] * input1[((rc * 2048) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 2048; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 2048) + ax3)] = max((((input3[((((ax1 * 7) + ax2) * 2048) + ax3)] + (Conv2dOutput[((ax2 * 2048) + ax3)] + input2[ax3])) * input4[ax3]) + input5[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fuse_mean_kernel0(__global float* restrict input0_red, __global float* restrict input0, __global float* restrict T_divide) {
  for (int ax3 = 0; ax3 < 2048; ++ax3) {
    input0_red[ax3] = 0.000000e+00f;
    for (int k1 = 0; k1 < 7; ++k1) {
      for (int k2 = 0; k2 < 7; ++k2) {
        input0_red[ax3] = (input0_red[ax3] + input0[((((k1 * 7) + k2) * 2048) + ax3)]);
      }
    }
  }
  for (int ax31 = 0; ax31 < 2048; ++ax31) {
    T_divide[ax31] = (input0_red[ax31] * 2.040816e-02f);
  }
}

__kernel void fuse_pad_15_kernel0(__global float* restrict T_pad, __global float* restrict input0) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 2048; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fuse_conv2d_broadcast_add_8_kernel0(__global float* restrict Conv2dOutput, __global float* restrict input0, __global float* restrict input1, __global float* restrict T_add, __global float* restrict input2) {
  for (int ff = 0; ff < 1001; ++ff) {
    Conv2dOutput[ff] = 0.000000e+00f;
    for (int rc = 0; rc < 2048; ++rc) {
      Conv2dOutput[ff] = (Conv2dOutput[ff] + (input0[rc] * input1[((rc * 1001) + ff)]));
    }
  }
  for (int ax3 = 0; ax3 < 1001; ++ax3) {
    T_add[ax3] = (Conv2dOutput[ax3] + input2[ax3]);
  }
}

__kernel void fuse_squeeze_reshape_flatten_kernel0(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    tensor[ax0_ax1_fused_inner] = input0[ax0_ax1_fused_inner];
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


