__kernel void fused_nn_conv2d_34_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 384; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 192; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (placeholder[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * placeholder1[((((((ff * 192) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_9_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 832; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (8 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (8 - rv1))) ? placeholder[(((((((ax1 * 7) + ax2) + rv) * 7) + ax3) + rv1) + -8)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_transpose_transpose_nn_pad_7_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_31_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 320; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 160; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (placeholder[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * placeholder1[((((((ff * 160) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_8_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 528; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? placeholder[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_transpose_transpose_nn_pad_6_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_30_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 528; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((rc * 14) + yy) * 14) + xx)] * placeholder1[((ff * 528) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_29_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 32; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * placeholder1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_32_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 32; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (placeholder[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * placeholder1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_27_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 32; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * placeholder1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_2_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 192; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((ax2 * 2) < (56 - rv)) && ((ax3 * 2) < (56 - rv1))) ? placeholder[((((((((ax1 * 28) + ax2) * 2) + rv) * 28) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_15_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 2592; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + 20376)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_pad_5_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_max_pool2d_7_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? placeholder[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_nn_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_transpose_transpose_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_multiply_add_nn_rel_12232970518842145914__2_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3, __global float* restrict placeholder4, __global float* restrict placeholder5, __global float* restrict placeholder6, __global float* restrict placeholder7, __global float* restrict placeholder8, __global float* restrict placeholder9, __global float* restrict placeholder10, __global float* restrict placeholder11) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 163072; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((137984 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -137984)] * placeholder1[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -704)]) + placeholder2[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -704)]), 0.000000e+00f) : (float)((112896 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -112896)] * placeholder4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -576)]) + placeholder5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -576)]), 0.000000e+00f) : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder6[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] * placeholder7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -256)]) + placeholder8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -256)]), 0.000000e+00f) : max(((placeholder9[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder10[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]) + placeholder11[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]), 0.000000e+00f))));
  }
}

__kernel void fused_transpose_transpose_2_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_multiply_add_nn_rel_12232970518842145914__6_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3, __global float* restrict placeholder4, __global float* restrict placeholder5, __global float* restrict placeholder6, __global float* restrict placeholder7, __global float* restrict placeholder8, __global float* restrict placeholder9, __global float* restrict placeholder10, __global float* restrict placeholder11) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] * placeholder1[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -448)]) + placeholder2[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -448)]), 0.000000e+00f) : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] * placeholder4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -400)]) + placeholder5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -400)]), 0.000000e+00f) : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder6[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] * placeholder7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -192)]) + placeholder8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -192)]), 0.000000e+00f) : max(((placeholder9[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder10[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]) + placeholder11[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]), 0.000000e+00f))));
  }
}

__kernel void fused_nn_conv2d_23_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 24; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * placeholder1[((((((ff * 24) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_multiply_add_nn_rel_12232970518842145914__3_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3, __global float* restrict placeholder4, __global float* restrict placeholder5, __global float* restrict placeholder6, __global float* restrict placeholder7, __global float* restrict placeholder8, __global float* restrict placeholder9, __global float* restrict placeholder10, __global float* restrict placeholder11) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((90944 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -90944)] * placeholder1[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -464)]) + placeholder2[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -464)]), 0.000000e+00f) : (float)((78400 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -78400)] * placeholder4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -400)]) + placeholder5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -400)]), 0.000000e+00f) : (float)((21952 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder6[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -21952)] * placeholder7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -112)]) + placeholder8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -112)]), 0.000000e+00f) : max(((placeholder9[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder10[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]) + placeholder11[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]), 0.000000e+00f))));
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_multiply_add_nn_rel_12232970518842145914__8_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3, __global float* restrict placeholder4, __global float* restrict placeholder5, __global float* restrict placeholder6, __global float* restrict placeholder7, __global float* restrict placeholder8, __global float* restrict placeholder9, __global float* restrict placeholder10, __global float* restrict placeholder11) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((175616 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -175616)] * placeholder1[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -224)]) + placeholder2[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -224)]), 0.000000e+00f) : (float)((150528 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -150528)] * placeholder4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -192)]) + placeholder5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -192)]), 0.000000e+00f) : (float)((50176 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder6[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -50176)] * placeholder7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -64)]) + placeholder8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -64)]), 0.000000e+00f) : max(((placeholder9[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder10[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + placeholder11[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f))));
  }
}

__kernel void fused_nn_conv2d_16_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 192; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (placeholder[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * placeholder1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_4_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 288; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((rc * 14) + yy) * 14) + xx)] * placeholder1[((ff * 512) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_1_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 624; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (placeholder[((((rc * 7) + yy) * 7) + xx)] * placeholder1[((ff * 832) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_5_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 280; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((rc * 14) + yy) * 14) + xx)] * placeholder1[((ff * 512) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_11_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 50161)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_10_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 36864; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 21937)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_avg_pool2d_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    tensor[ax1] = 0.000000e+00f;
    for (int rv = 0; rv < 7; ++rv) {
      for (int rv1 = 0; rv1 < 7; ++rv1) {
        tensor[ax1] = (tensor[ax1] + (placeholder[((((ax1 * 7) + rv) * 7) + rv1)] * 2.040816e-02f));
      }
    }
  }
}

__kernel void fused_nn_conv2d_6_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 296; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((rc * 14) + yy) * 14) + xx)] * placeholder1[((ff * 512) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_12_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40960; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 50161)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_reshape_kernel0(__global float* restrict T_reshape, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    T_reshape[ax0_ax1_fused_inner] = placeholder[ax0_ax1_fused_inner];
  }
}

__kernel void fused_transpose_transpose_nn_pad_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1024; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_21_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((rc * 14) + yy) * 14) + xx)] * placeholder1[((ff * 480) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_8_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 288; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (placeholder[((((rc * 28) + yy) * 28) + xx)] * placeholder1[((ff * 256) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_33_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (placeholder[((((rc * 7) + yy) * 7) + xx)] * placeholder1[((ff * 832) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_transpose_transpose_nn_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_19_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 208; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 96; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * placeholder1[((((((ff * 96) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_softmax_kernel0(__global float* restrict tensor, __global float* restrict placeholder, __global float* restrict tensor1, __global float* restrict tensor2) {
  for (int ax1 = 0; ax1 < 1001; ++ax1) {
    tensor[0] = -3.402823e+38f;
    for (int k1 = 0; k1 < 1001; ++k1) {
      tensor[0] = max(tensor[0], placeholder[k1]);
    }
    tensor1[0] = 0.000000e+00f;
    for (int k2 = 0; k2 < 1001; ++k2) {
      tensor1[0] = (tensor1[0] + exp((placeholder[k2] - tensor[0])));
    }
    tensor2[ax1] = (exp((placeholder[ax1] - tensor[0])) / tensor1[0]);
  }
}

__kernel void fused_nn_conv2d_9_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 176; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (placeholder[((((rc * 28) + yy) * 28) + xx)] * placeholder1[((ff * 192) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 94080; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_multiply_add_nn_rel_12232970518842145914__4_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3, __global float* restrict placeholder4, __global float* restrict placeholder5, __global float* restrict placeholder6, __global float* restrict placeholder7, __global float* restrict placeholder8, __global float* restrict placeholder9, __global float* restrict placeholder10, __global float* restrict placeholder11) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] * placeholder1[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -448)]) + placeholder2[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -448)]), 0.000000e+00f) : (float)((75264 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -75264)] * placeholder4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -384)]) + placeholder5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -384)]), 0.000000e+00f) : (float)((25088 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder6[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -25088)] * placeholder7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -128)]) + placeholder8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -128)]), 0.000000e+00f) : max(((placeholder9[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder10[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]) + placeholder11[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]), 0.000000e+00f))));
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 24576; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 37617)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_pad_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_14_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 12960; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + 12536)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_2_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 448; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 832; ++rc) {
          compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (placeholder[((((rc * 7) + yy) * 7) + xx)] * placeholder1[((ff * 832) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_multiply_add_nn_rel_12232970518842145914__5_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3, __global float* restrict placeholder4, __global float* restrict placeholder5, __global float* restrict placeholder6, __global float* restrict placeholder7, __global float* restrict placeholder8, __global float* restrict placeholder9, __global float* restrict placeholder10, __global float* restrict placeholder11) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((87808 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -87808)] * placeholder1[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -448)]) + placeholder2[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -448)]), 0.000000e+00f) : (float)((75264 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -75264)] * placeholder4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -384)]) + placeholder5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -384)]), 0.000000e+00f) : (float)((31360 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder6[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -31360)] * placeholder7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -160)]) + placeholder8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196) + -160)]), 0.000000e+00f) : max(((placeholder9[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder10[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]) + placeholder11[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 196)]), 0.000000e+00f))));
  }
}

__kernel void fused_nn_max_pool2d_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 832; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 7; ++ax3) {
        tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 2; ++rv) {
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tensor[((((ax1 * 7) + ax2) * 7) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 7) + ax3)], placeholder[((((((((ax1 * 7) + ax2) * 2) + rv) * 7) + ax3) * 2) + rv1)]);
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_17_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 3888; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + 28216)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_28_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 320; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 160; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * placeholder1[((((((ff * 160) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_multiply_add_nn_rel_12232970518842145914__7_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3, __global float* restrict placeholder4, __global float* restrict placeholder5, __global float* restrict placeholder6, __global float* restrict placeholder7, __global float* restrict placeholder8, __global float* restrict placeholder9, __global float* restrict placeholder10, __global float* restrict placeholder11) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 376320; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((326144 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -326144)] * placeholder1[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -416)]) + placeholder2[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -416)]), 0.000000e+00f) : (float)((250880 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -250880)] * placeholder4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -320)]) + placeholder5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -320)]), 0.000000e+00f) : (float)((100352 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder6[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -100352)] * placeholder7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -128)]) + placeholder8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784) + -128)]), 0.000000e+00f) : max(((placeholder9[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder10[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + placeholder11[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f))));
  }
}

__kernel void fused_transpose_multiply_add_nn_relu_transpose_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 602112; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]), 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_14_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 16; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (placeholder[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * placeholder1[((((((ff * 16) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_10_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 192; ++ff) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        compute[((((ff * 56) + yy) * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (placeholder[((((((rc * 58) + yy) + ry) * 58) + xx) + rx)] * placeholder1[((((((ff * 64) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_transpose_multiply_add_nn_relu_transpose_nn_pad_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_11_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 56; ++yy) {
      for (int xx = 0; xx < 56; ++xx) {
        compute[((((ff * 56) + yy) * 56) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (placeholder[((((rc * 56) + yy) * 56) + xx)] * placeholder1[((ff * 64) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_7_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 304; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 480; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((rc * 14) + yy) * 14) + xx)] * placeholder1[((ff * 480) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_3_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 64; ++ax1) {
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 56; ++ax3) {
        tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = max(tensor[((((ax1 * 56) + ax2) * 56) + ax3)], (float)((((ax2 * 2) < (112 - rv)) && ((ax3 * 2) < (112 - rv1))) ? placeholder[((((((((ax1 * 56) + ax2) * 2) + rv) * 56) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_transpose_multiply_add_nn_relu_transpose_1_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 12544)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 12544)]), 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 14400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + 125411)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_transpose_transpose_nn_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_max_pool2d_6_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 480; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (15 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (15 - rv1))) ? placeholder[(((((((ax1 * 14) + ax2) + rv) * 14) + ax3) + rv1) + -15)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_16_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 15552; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((9 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) < 72)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9) < 8)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 81) / 9)) * 7) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 9)) + 18808)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 81)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_multiply_add_nn_rel_12232970518842145914__kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3, __global float* restrict placeholder4, __global float* restrict placeholder5, __global float* restrict placeholder6, __global float* restrict placeholder7, __global float* restrict placeholder8, __global float* restrict placeholder9, __global float* restrict placeholder10, __global float* restrict placeholder11) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((43904 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -43904)] * placeholder1[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -896)]) + placeholder2[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -896)]), 0.000000e+00f) : (float)((37632 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -37632)] * placeholder4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -768)]) + placeholder5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -768)]), 0.000000e+00f) : (float)((18816 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder6[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -18816)] * placeholder7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -384)]) + placeholder8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -384)]), 0.000000e+00f) : max(((placeholder9[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder10[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + placeholder11[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f))));
  }
}

__kernel void fused_nn_conv2d_17_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 96; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 32; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (placeholder[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * placeholder1[((((((ff * 32) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_transpose_nn_pad_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 157323; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((458 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) < 51754)) && (2 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229) < 226)) ? placeholder[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) / 229) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52441)) + -1350)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_13_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 96; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (placeholder[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * placeholder1[((((((ff * 96) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_1_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 480; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], (float)((((ax2 * 2) < (28 - rv)) && ((ax3 * 2) < (28 - rv1))) ? placeholder[((((((((ax1 * 14) + ax2) * 2) + rv) * 14) + ax3) * 2) + rv1)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_22_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 224; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 112; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * placeholder1[((((((ff * 112) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_multiply_add_nn_rel_12232970518842145914__1_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3, __global float* restrict placeholder4, __global float* restrict placeholder5, __global float* restrict placeholder6, __global float* restrict placeholder7, __global float* restrict placeholder8, __global float* restrict placeholder9, __global float* restrict placeholder10, __global float* restrict placeholder11) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((34496 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -34496)] * placeholder1[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -704)]) + placeholder2[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -704)]), 0.000000e+00f) : (float)((28224 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder3[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -28224)] * placeholder4[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -576)]) + placeholder5[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -576)]), 0.000000e+00f) : (float)((12544 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) ? max(((placeholder6[(ax0_ax1_fused_ax2_fused_ax3_fused_inner + -12544)] * placeholder7[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -256)]) + placeholder8[((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49) + -256)]), 0.000000e+00f) : max(((placeholder9[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * placeholder10[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]) + placeholder11[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 49)]), 0.000000e+00f))));
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 86400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + 50147)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_15_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 32; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 192; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (placeholder[((((rc * 28) + yy) * 28) + xx)] * placeholder1[((ff * 192) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_4_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 192; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? placeholder[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + 100323)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28800; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + 200675)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_18_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (placeholder[((((rc * 28) + yy) * 28) + xx)] * placeholder1[((ff * 256) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_5_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 256; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((((1 - rv) <= ax2) && (ax2 < (29 - rv))) && ((1 - rv1) <= ax3)) && (ax3 < (29 - rv1))) ? placeholder[(((((((ax1 * 28) + ax2) + rv) * 28) + ax3) + rv1) + -29)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_13_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 81521)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_6_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28672; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 31345)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_9_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 50161)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 1001; ++ff) {
    compute[ff] = 0.000000e+00f;
    for (int rc = 0; rc < 1024; ++rc) {
      compute[ff] = (compute[ff] + (placeholder[rc] * placeholder1[((ff * 1024) + rc)]));
    }
  }
}

__kernel void fused_transpose_transpose_nn_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_20_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 48; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 16; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * placeholder1[((((((ff * 16) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_5_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 4096; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 56433)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_transpose_transpose_1_kernel0(__global float* restrict T_transpose, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 94080; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_12_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 112; ++yy) {
      for (int xx = 0; xx < 112; ++xx) {
        compute[((((ff * 112) + yy) * 112) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 3; ++rc) {
          for (int ry = 0; ry < 7; ++ry) {
            for (int rx = 0; rx < 7; ++rx) {
              compute[((((ff * 112) + yy) * 112) + xx)] = (compute[((((ff * 112) + yy) * 112) + xx)] + (placeholder[(((((rc * 52441) + (yy * 458)) + (ry * 229)) + (xx * 2)) + rx)] * placeholder1[((((((ff * 3) + rc) * 7) + ry) * 7) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_transpose_transpose_nn_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 94080; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_7_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 53297)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_3_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 448; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 528; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((rc * 14) + yy) * 14) + xx)] * placeholder1[((ff * 528) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_24_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 64; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((rc * 14) + yy) * 14) + xx)] * placeholder1[((ff * 512) + rc)]));
        }
      }
    }
  }
}

__kernel void fused_transpose_add_squeeze_reshape_kernel0(__global float* restrict T_reshape, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    T_reshape[ax0_ax1_fused_inner] = (placeholder[ax0_ax1_fused_inner] + placeholder1[ax0_ax1_fused_inner]);
  }
}

__kernel void fused_transpose_transpose_nn_pad_5_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_25_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 256; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 128; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * placeholder1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_transpose_multiply_add_nn_relu_transpose_nn_pad_8_kernel0(__global float* restrict T_pad, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 32768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16) < 15)) ? max(((placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 14) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + 25073)] * placeholder1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]) + placeholder2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256)]), 0.000000e+00f) : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_35_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int xx = 0; xx < 7; ++xx) {
        compute[((((ff * 7) + yy) * 7) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 48; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 7) + yy) * 7) + xx)] = (compute[((((ff * 7) + yy) * 7) + xx)] + (placeholder[((((((rc * 9) + yy) + ry) * 9) + xx) + rx)] * placeholder1[((((((ff * 48) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_26_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  for (int ff = 0; ff < 288; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 144; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (placeholder[((((((rc * 16) + yy) + ry) * 16) + xx) + rx)] * placeholder1[((((((ff * 144) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
      }
    }
  }
}

