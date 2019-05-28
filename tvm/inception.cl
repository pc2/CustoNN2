__kernel void fused_nn_conv2d_add_nn_relu_34_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 48; ++rc) {
              Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (placeholder[((((((ax1 + ry) * 9) + xx) + rx) * 48) + rc)] * placeholder1[((((((ry * 3) + rx) * 48) + rc) * 128) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 128) + ax3)] = max((Conv2dOutput[((ax2 * 128) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_32_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 832; ++rc) {
          Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (placeholder[((((ax1 * 7) + xx) * 832) + rc)] * placeholder1[((rc * 128) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 128) + ax3)] = max((Conv2dOutput[((ax2 * 128) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_15_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 2592; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((288 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 2304)) && (32 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 288))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 288) < 256)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 288) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 288) / 32)) * 448) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 32)) + -3168)] : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_nn_pad_14_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 12960; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((1440 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 11520)) && (160 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1440))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1440) < 1280)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1440) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1440) / 160)) * 448) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 160)) + -3328)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_31_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 32; ++rc) {
              Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (placeholder[((((((ax1 + ry) * 9) + xx) + rx) * 32) + rc)] * placeholder1[((((((ry * 3) + rx) * 32) + rc) * 128) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 128) + ax3)] = max((Conv2dOutput[((ax2 * 128) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_concatenate_1_kernel0(__global float* restrict T_concat, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((704 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832) * 128) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) + -704)] : (float)((576 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832) * 128) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) + -576)] : (float)((256 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? placeholder2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832) * 320) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) + -256)] : placeholder3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832) * 448) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832))])));
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_29_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 528; ++rc) {
          Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (placeholder[((((ax1 * 14) + xx) * 528) + rc)] * placeholder1[((rc * 128) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 128) + ax3)] = max((Conv2dOutput[((ax2 * 128) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_9_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int xx = 0; xx < 56; ++xx) {
      for (int ff = 0; ff < 192; ++ff) {
        Conv2dOutput[((xx * 192) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 64; ++rc) {
              Conv2dOutput[((xx * 192) + ff)] = (Conv2dOutput[((xx * 192) + ff)] + (placeholder[((((((ax1 + ry) * 58) + xx) + rx) * 64) + rc)] * placeholder1[((((((ry * 3) + rx) * 64) + rc) * 192) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 192; ++ax3) {
        T_relu[((((ax1 * 56) + ax2) * 192) + ax3)] = max((Conv2dOutput[((ax2 * 192) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_5_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 4096; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((256 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 3840)) && (16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) < 240)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256) / 16)) * 304) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -4272)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_8_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 176; ++ff) {
        Conv2dOutput[((xx * 176) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 192; ++rc) {
          Conv2dOutput[((xx * 176) + ff)] = (Conv2dOutput[((xx * 176) + ff)] + (placeholder[((((ax1 * 28) + xx) * 192) + rc)] * placeholder1[((rc * 176) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 176; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 176) + ax3)] = max((Conv2dOutput[((ax2 * 176) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_pad_7_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((3712 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 211584)) && (64 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3712))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3712) < 3648)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3712) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3712)) + -3648)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_max_pool2d_2_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 192; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 192) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 192) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 192) + ax3)], (float)((((ax1 * 2) < (56 - rv)) && ((ax2 * 2) < (56 - rv1))) ? placeholder[((((((((ax1 * 2) + rv) * 28) + ax2) * 2) + rv1) * 192) + ax3)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_squeeze_reshape_kernel0(__global float* restrict T_reshape, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    T_reshape[ax0_ax1_fused_inner] = placeholder[ax0_ax1_fused_inner];
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_10_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int xx = 0; xx < 56; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 64; ++rc) {
          Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (placeholder[((((ax1 * 56) + xx) * 64) + rc)] * placeholder1[((rc * 64) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 56) + ax2) * 64) + ax3)] = max((Conv2dOutput[((ax2 * 64) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_pad_5_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_30_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 320; ++ff) {
        Conv2dOutput[((xx * 320) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 160; ++rc) {
              Conv2dOutput[((xx * 320) + ff)] = (Conv2dOutput[((xx * 320) + ff)] + (placeholder[((((((ax1 + ry) * 9) + xx) + rx) * 160) + rc)] * placeholder1[((((((ry * 3) + rx) * 160) + rc) * 320) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 320; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 320) + ax3)] = max((Conv2dOutput[((ax2 * 320) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_concatenate_7_kernel0(__global float* restrict T_concat, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 376320; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((416 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 480) * 64) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480)) + -416)] : (float)((320 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480)) ? placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 480) * 96) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480)) + -320)] : (float)((128 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480)) ? placeholder2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 480) * 192) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480)) + -128)] : placeholder3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 480) * 288) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480))])));
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_24_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 256; ++ff) {
        Conv2dOutput[((xx * 256) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 128; ++rc) {
              Conv2dOutput[((xx * 256) + ff)] = (Conv2dOutput[((xx * 256) + ff)] + (placeholder[((((((ax1 + ry) * 16) + xx) + rx) * 128) + rc)] * placeholder1[((((((ry * 3) + rx) * 128) + rc) * 256) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 256) + ax3)] = max((Conv2dOutput[((ax2 * 256) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_6_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 480; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 480) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 480) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 480) + ax3)], (float)((((((1 - rv) <= ax1) && (ax1 < (15 - rv))) && ((1 - rv1) <= ax2)) && (ax2 < (15 - rv1))) ? placeholder[(((((((ax1 + rv) * 14) + ax2) + rv1) * 480) + ax3) + -7200)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_concatenate_6_kernel0(__global float* restrict T_concat, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((448 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 64) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) + -448)] : (float)((400 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 48) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) + -400)] : (float)((192 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? placeholder2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 208) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) + -192)] : placeholder3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 304) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512))])));
  }
}

__kernel void fused_nn_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 94080; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_strided_slice_nn_pad_4_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 24576; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((1536 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 23040)) && (96 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1536))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1536) < 1440)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1536) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1536) / 96)) * 304) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 96)) + -4368)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_5_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 296; ++ff) {
        Conv2dOutput[((xx * 296) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 296) + ff)] = (Conv2dOutput[((xx * 296) + ff)] + (placeholder[((((ax1 * 14) + xx) * 512) + rc)] * placeholder1[((rc * 296) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 296; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 296) + ax3)] = max((Conv2dOutput[((ax2 * 296) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_concatenate_8_kernel0(__global float* restrict T_concat, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((224 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 32) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) + -224)] : (float)((192 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) ? placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 32) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) + -192)] : (float)((64 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) ? placeholder2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 128) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256)) + -64)] : placeholder3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 256) * 176) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 256))])));
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_28_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 32; ++rc) {
              Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (placeholder[((((((ax1 + ry) * 16) + xx) + rx) * 32) + rc)] * placeholder1[((((((ry * 3) + rx) * 32) + rc) * 128) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 128) + ax3)] = max((Conv2dOutput[((ax2 * 128) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_avg_pool2d_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax3 = 0; ax3 < 1024; ++ax3) {
    tensor[ax3] = 0.000000e+00f;
    for (int rv = 0; rv < 7; ++rv) {
      for (int rv1 = 0; rv1 < 7; ++rv1) {
        tensor[ax3] = (tensor[ax3] + (placeholder[((((rv * 7) + rv1) * 1024) + ax3)] * 2.040816e-02f));
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_4_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 280; ++ff) {
        Conv2dOutput[((xx * 280) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 280) + ff)] = (Conv2dOutput[((xx * 280) + ff)] + (placeholder[((((ax1 * 14) + xx) * 512) + rc)] * placeholder1[((rc * 280) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 280; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 280) + ax3)] = max((Conv2dOutput[((ax2 * 280) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_16_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 15552; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((1728 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 13824)) && (192 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1728))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1728) < 1536)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1728) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1728) / 192)) * 624) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 192)) + -4608)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_add_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_add, __global float* restrict placeholder2) {
  for (int ff = 0; ff < 1001; ++ff) {
    Conv2dOutput[ff] = 0.000000e+00f;
    for (int rc = 0; rc < 1024; ++rc) {
      Conv2dOutput[ff] = (Conv2dOutput[ff] + (placeholder[rc] * placeholder1[((rc * 1001) + ff)]));
    }
  }
  for (int ax3 = 0; ax3 < 1001; ++ax3) {
    T_add[ax3] = (Conv2dOutput[ax3] + placeholder2[ax3]);
  }
}

__kernel void fused_nn_pad_6_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_pad_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1024; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_3_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 288; ++ff) {
        Conv2dOutput[((xx * 288) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 288) + ff)] = (Conv2dOutput[((xx * 288) + ff)] + (placeholder[((((ax1 * 14) + xx) * 512) + rc)] * placeholder1[((rc * 288) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 288; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 288) + ax3)] = max((Conv2dOutput[((ax2 * 288) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28800; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((960 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 27840)) && (32 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 960))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 960) < 928)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 960) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 960) / 32)) * 288) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 32)) + -8096)] : 0.000000e+00f);
  }
}

__kernel void fused_reshape_kernel0(__global float* restrict T_reshape, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 1001; ++ax0_ax1_fused_inner) {
    T_reshape[ax0_ax1_fused_inner] = placeholder[ax0_ax1_fused_inner];
  }
}

__kernel void fused_strided_slice_concatenate_kernel0(__global float* restrict T_concat, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 50176; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((896 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1024) * 128) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)) + -896)] : (float)((768 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)) ? placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1024) * 128) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)) + -768)] : (float)((384 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)) ? placeholder2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1024) * 384) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024)) + -384)] : placeholder3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1024) * 624) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1024))])));
  }
}

__kernel void fused_nn_max_pool2d_9_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 832; ++ax3) {
        tensor[((((ax1 * 7) + ax2) * 832) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 7) + ax2) * 832) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 832) + ax3)], (float)((((((1 - rv) <= ax1) && (ax1 < (8 - rv))) && ((1 - rv1) <= ax2)) && (ax2 < (8 - rv1))) ? placeholder[(((((((ax1 + rv) * 7) + ax2) + rv1) * 832) + ax3) + -6656)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_1_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 448; ++ff) {
        Conv2dOutput[((xx * 448) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 832; ++rc) {
          Conv2dOutput[((xx * 448) + ff)] = (Conv2dOutput[((xx * 448) + ff)] + (placeholder[((((ax1 * 7) + xx) * 832) + rc)] * placeholder1[((rc * 448) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 448; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 448) + ax3)] = max((Conv2dOutput[((ax2 * 448) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_concatenate_3_kernel0(__global float* restrict T_concat, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((464 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 528) * 64) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528)) + -464)] : (float)((400 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528)) ? placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 528) * 64) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528)) + -400)] : (float)((112 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528)) ? placeholder2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 528) * 288) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528)) + -112)] : placeholder3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 528) * 288) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 528))])));
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_33_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 384; ++ff) {
        Conv2dOutput[((xx * 384) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 192; ++rc) {
              Conv2dOutput[((xx * 384) + ff)] = (Conv2dOutput[((xx * 384) + ff)] + (placeholder[((((((ax1 + ry) * 9) + xx) + rx) * 192) + rc)] * placeholder1[((((((ry * 3) + rx) * 192) + rc) * 384) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 384; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 384) + ax3)] = max((Conv2dOutput[((ax2 * 384) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int xx = 0; xx < 7; ++xx) {
      for (int ff = 0; ff < 624; ++ff) {
        Conv2dOutput[((xx * 624) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 832; ++rc) {
          Conv2dOutput[((xx * 624) + ff)] = (Conv2dOutput[((xx * 624) + ff)] + (placeholder[((((ax1 * 7) + xx) * 832) + rc)] * placeholder1[((rc * 624) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 624; ++ax3) {
        T_relu[((((ax1 * 7) + ax2) * 624) + ax3)] = max((Conv2dOutput[((ax2 * 624) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_strided_slice_nn_pad_8_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 32768; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((2048 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 30720)) && (128 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048) < 1920)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 2048) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2048) / 128)) * 280) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 128)) + -4072)] : 0.000000e+00f);
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

__kernel void fused_nn_max_pool2d_4_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 192; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 192) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 192) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 192) + ax3)], (float)((((((1 - rv) <= ax1) && (ax1 < (29 - rv))) && ((1 - rv1) <= ax2)) && (ax2 < (29 - rv1))) ? placeholder[(((((((ax1 + rv) * 28) + ax2) + rv1) * 192) + ax3) + -5568)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 7; ++ax1) {
    for (int ax2 = 0; ax2 < 7; ++ax2) {
      for (int ax3 = 0; ax3 < 832; ++ax3) {
        tensor[((((ax1 * 7) + ax2) * 832) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 2; ++rv) {
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tensor[((((ax1 * 7) + ax2) * 832) + ax3)] = max(tensor[((((ax1 * 7) + ax2) * 832) + ax3)], placeholder[((((((((ax1 * 2) + rv) * 7) + ax2) * 2) + rv1) * 832) + ax3)]);
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_concatenate_2_kernel0(__global float* restrict T_concat, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 163072; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((704 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832) * 128) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) + -704)] : (float)((576 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832) * 128) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) + -576)] : (float)((256 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) ? placeholder2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832) * 320) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832)) + -256)] : placeholder3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 832) * 448) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 832))])));
  }
}

__kernel void fused_nn_max_pool2d_1_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 480; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 480) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 480) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 480) + ax3)], (float)((((ax1 * 2) < (28 - rv)) && ((ax2 * 2) < (28 - rv1))) ? placeholder[((((((((ax1 * 2) + rv) * 14) + ax2) * 2) + rv1) * 480) + ax3)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_2_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 448; ++ff) {
        Conv2dOutput[((xx * 448) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 528; ++rc) {
          Conv2dOutput[((xx * 448) + ff)] = (Conv2dOutput[((xx * 448) + ff)] + (placeholder[((((ax1 * 14) + xx) * 528) + rc)] * placeholder1[((rc * 448) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 448; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 448) + ax3)] = max((Conv2dOutput[((ax2 * 448) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_pad_8_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_6_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 304; ++ff) {
        Conv2dOutput[((xx * 304) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 480; ++rc) {
          Conv2dOutput[((xx * 304) + ff)] = (Conv2dOutput[((xx * 304) + ff)] + (placeholder[((((ax1 * 14) + xx) * 480) + rc)] * placeholder1[((rc * 304) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 304; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 304) + ax3)] = max((Conv2dOutput[((ax2 * 304) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_27_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 320; ++ff) {
        Conv2dOutput[((xx * 320) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 160; ++rc) {
              Conv2dOutput[((xx * 320) + ff)] = (Conv2dOutput[((xx * 320) + ff)] + (placeholder[((((((ax1 + ry) * 16) + xx) + rx) * 160) + rc)] * placeholder1[((((((ry * 3) + rx) * 160) + rc) * 320) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 320; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 320) + ax3)] = max((Conv2dOutput[((ax2 * 320) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_11_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 112; ++ax1) {
    for (int xx = 0; xx < 112; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 7; ++ry) {
          for (int rx = 0; rx < 7; ++rx) {
            for (int rc = 0; rc < 3; ++rc) {
              Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (placeholder[((((((ax1 * 2) + ry) * 687) + (xx * 6)) + (rx * 3)) + rc)] * placeholder1[((((((ry * 7) + rx) * 3) + rc) * 64) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 112; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 112) + ax2) * 64) + ax3)] = max((Conv2dOutput[((ax2 * 64) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_17_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 3888; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((432 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 3456)) && (48 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 432))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 432) < 384)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 432) * 7) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 432) / 48)) * 624) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 48)) + -4416)] : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_nn_pad_11_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((512 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 7680)) && (32 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) < 480)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) / 32)) * 288) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 32)) + -4064)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_pad_9_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 157323; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((1374 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 155262)) && (6 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 687))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 687) < 678)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 687) * 672) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 687)) + -1350)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_max_pool2d_3_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 56; ++ax1) {
    for (int ax2 = 0; ax2 < 56; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        tensor[((((ax1 * 56) + ax2) * 64) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 56) + ax2) * 64) + ax3)] = max(tensor[((((ax1 * 56) + ax2) * 64) + ax3)], (float)((((ax1 * 2) < (112 - rv)) && ((ax2 * 2) < (112 - rv1))) ? placeholder[((((((((ax1 * 2) + rv) * 56) + ax2) * 2) + rv1) * 64) + ax3)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((3840 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 111360)) && (128 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3840))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3840) < 3712)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3840) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3840) / 128)) * 288) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 128)) + -8224)] : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_nn_pad_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 86400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((2880 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 83520)) && (96 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2880))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2880) < 2784)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 2880) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2880) / 96)) * 176) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 96)) + -5040)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_pad_2_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 103488; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_13_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 32; ++ff) {
        Conv2dOutput[((xx * 32) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 16; ++rc) {
              Conv2dOutput[((xx * 32) + ff)] = (Conv2dOutput[((xx * 32) + ff)] + (placeholder[((((((ax1 + ry) * 30) + xx) + rx) * 16) + rc)] * placeholder1[((((((ry * 3) + rx) * 16) + rc) * 32) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 32; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 32) + ax3)] = max((Conv2dOutput[((ax2 * 32) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_1_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 14400; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((480 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 13920)) && (16 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480) < 464)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 480) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 480) / 16)) * 176) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 16)) + -4944)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_14_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 32; ++ff) {
        Conv2dOutput[((xx * 32) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 192; ++rc) {
          Conv2dOutput[((xx * 32) + ff)] = (Conv2dOutput[((xx * 32) + ff)] + (placeholder[((((ax1 * 28) + xx) * 192) + rc)] * placeholder1[((rc * 32) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 32; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 32) + ax3)] = max((Conv2dOutput[((ax2 * 32) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_19_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 48; ++ff) {
        Conv2dOutput[((xx * 48) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 16; ++rc) {
              Conv2dOutput[((xx * 48) + ff)] = (Conv2dOutput[((xx * 48) + ff)] + (placeholder[((((((ax1 + ry) * 16) + xx) + rx) * 16) + rc)] * placeholder1[((((((ry * 3) + rx) * 16) + rc) * 48) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 48; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 48) + ax3)] = max((Conv2dOutput[((ax2 * 48) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_15_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 192; ++ff) {
        Conv2dOutput[((xx * 192) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 128; ++rc) {
              Conv2dOutput[((xx * 192) + ff)] = (Conv2dOutput[((xx * 192) + ff)] + (placeholder[((((((ax1 + ry) * 30) + xx) + rx) * 128) + rc)] * placeholder1[((((((ry * 3) + rx) * 128) + rc) * 192) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 192; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 192) + ax3)] = max((Conv2dOutput[((ax2 * 192) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_pad_3_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = placeholder[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_16_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 96; ++ff) {
        Conv2dOutput[((xx * 96) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 32; ++rc) {
              Conv2dOutput[((xx * 96) + ff)] = (Conv2dOutput[((xx * 96) + ff)] + (placeholder[((((((ax1 + ry) * 30) + xx) + rx) * 32) + rc)] * placeholder1[((((((ry * 3) + rx) * 32) + rc) * 96) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 96; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 96) + ax3)] = max((Conv2dOutput[((ax2 * 96) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_7_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((384 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 5760)) && (24 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 384))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 384) < 360)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 384) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 384) / 24)) * 296) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 24)) + -4168)] : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_concatenate_4_kernel0(__global float* restrict T_concat, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((448 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 64) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) + -448)] : (float)((384 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 64) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) + -384)] : (float)((128 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? placeholder2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 256) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) + -128)] : placeholder3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 280) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512))])));
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_17_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (placeholder[((((ax1 * 28) + xx) * 256) + rc)] * placeholder1[((rc * 64) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 64) + ax3)] = max((Conv2dOutput[((ax2 * 64) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_12_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 128; ++ff) {
        Conv2dOutput[((xx * 128) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 96; ++rc) {
              Conv2dOutput[((xx * 128) + ff)] = (Conv2dOutput[((xx * 128) + ff)] + (placeholder[((((((ax1 + ry) * 30) + xx) + rx) * 96) + rc)] * placeholder1[((((((ry * 3) + rx) * 96) + rc) * 128) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 128; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 128) + ax3)] = max((Conv2dOutput[((ax2 * 128) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_5_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 256; ++ax3) {
        tensor[((((ax1 * 28) + ax2) * 256) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 28) + ax2) * 256) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 256) + ax3)], (float)((((((1 - rv) <= ax1) && (ax1 < (29 - rv))) && ((1 - rv1) <= ax2)) && (ax2 < (29 - rv1))) ? placeholder[(((((((ax1 + rv) * 28) + ax2) + rv1) * 256) + ax3) + -7424)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_13_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8192; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((512 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 7680)) && (32 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) < 480)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512) / 32)) * 448) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 32)) + -6304)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_18_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 208; ++ff) {
        Conv2dOutput[((xx * 208) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 96; ++rc) {
              Conv2dOutput[((xx * 208) + ff)] = (Conv2dOutput[((xx * 208) + ff)] + (placeholder[((((((ax1 + ry) * 16) + xx) + rx) * 96) + rc)] * placeholder1[((((((ry * 3) + rx) * 96) + rc) * 208) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 208; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 208) + ax3)] = max((Conv2dOutput[((ax2 * 208) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_concatenate_5_kernel0(__global float* restrict T_concat, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict placeholder2, __global float* restrict placeholder3) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_concat[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)((448 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 64) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) + -448)] : (float)((384 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? placeholder1[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 64) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) + -384)] : (float)((160 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) ? placeholder2[((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512)) + -160)] : placeholder3[(((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 512) * 296) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 512))])));
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_20_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 480; ++rc) {
          Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (placeholder[((((ax1 * 14) + xx) * 480) + rc)] * placeholder1[((rc * 64) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 64) + ax3)] = max((Conv2dOutput[((ax2 * 64) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_21_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 224; ++ff) {
        Conv2dOutput[((xx * 224) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 112; ++rc) {
              Conv2dOutput[((xx * 224) + ff)] = (Conv2dOutput[((xx * 224) + ff)] + (placeholder[((((((ax1 + ry) * 16) + xx) + rx) * 112) + rc)] * placeholder1[((((((ry * 3) + rx) * 112) + rc) * 224) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 224; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 224) + ax3)] = max((Conv2dOutput[((ax2 * 224) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_6_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 28672; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((1792 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 26880)) && (112 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1792))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1792) < 1680)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 1792) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 1792) / 112)) * 296) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 112)) + -4280)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_22_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 24; ++rc) {
              Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (placeholder[((((((ax1 + ry) * 16) + xx) + rx) * 24) + rc)] * placeholder1[((((((ry * 3) + rx) * 24) + rc) * 64) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 64) + ax3)] = max((Conv2dOutput[((ax2 * 64) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_23_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 512; ++rc) {
          Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (placeholder[((((ax1 * 14) + xx) * 512) + rc)] * placeholder1[((rc * 64) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 64) + ax3)] = max((Conv2dOutput[((ax2 * 64) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_max_pool2d_7_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 512; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 512) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 512) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 512) + ax3)], (float)((((((1 - rv) <= ax1) && (ax1 < (15 - rv))) && ((1 - rv1) <= ax2)) && (ax2 < (15 - rv1))) ? placeholder[(((((((ax1 + rv) * 14) + ax2) + rv1) * 512) + ax3) + -7680)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_26_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 64; ++ff) {
        Conv2dOutput[((xx * 64) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 32; ++rc) {
              Conv2dOutput[((xx * 64) + ff)] = (Conv2dOutput[((xx * 64) + ff)] + (placeholder[((((((ax1 + ry) * 16) + xx) + rx) * 32) + rc)] * placeholder1[((((((ry * 3) + rx) * 32) + rc) * 64) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 64; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 64) + ax3)] = max((Conv2dOutput[((ax2 * 64) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_9_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 6144; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((384 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 5760)) && (24 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 384))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 384) < 360)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 384) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 384) / 24)) * 280) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 24)) + -3944)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_25_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int xx = 0; xx < 14; ++xx) {
      for (int ff = 0; ff < 288; ++ff) {
        Conv2dOutput[((xx * 288) + ff)] = 0.000000e+00f;
        for (int ry = 0; ry < 3; ++ry) {
          for (int rx = 0; rx < 3; ++rx) {
            for (int rc = 0; rc < 144; ++rc) {
              Conv2dOutput[((xx * 288) + ff)] = (Conv2dOutput[((xx * 288) + ff)] + (placeholder[((((((ax1 + ry) * 16) + xx) + rx) * 144) + rc)] * placeholder1[((((((ry * 3) + rx) * 144) + rc) * 288) + ff)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 288; ++ax3) {
        T_relu[((((ax1 * 14) + ax2) * 288) + ax3)] = max((Conv2dOutput[((ax2 * 288) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_conv2d_add_nn_relu_7_kernel0(__global float* restrict Conv2dOutput, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu, __global float* restrict placeholder2) {
  for (int ax1 = 0; ax1 < 28; ++ax1) {
    for (int xx = 0; xx < 28; ++xx) {
      for (int ff = 0; ff < 288; ++ff) {
        Conv2dOutput[((xx * 288) + ff)] = 0.000000e+00f;
        for (int rc = 0; rc < 256; ++rc) {
          Conv2dOutput[((xx * 288) + ff)] = (Conv2dOutput[((xx * 288) + ff)] + (placeholder[((((ax1 * 28) + xx) * 256) + rc)] * placeholder1[((rc * 288) + ff)]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 288; ++ax3) {
        T_relu[((((ax1 * 28) + ax2) * 288) + ax3)] = max((Conv2dOutput[((ax2 * 288) + ax3)] + placeholder2[ax3]), 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_strided_slice_nn_pad_10_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 36864; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((2304 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 34560)) && (144 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2304))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2304) < 2160)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 2304) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2304) / 144)) * 288) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 144)) + -4208)] : 0.000000e+00f);
  }
}

__kernel void fused_strided_slice_nn_pad_12_kernel0(__global float* restrict T_pad, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 40960; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((2560 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner < 38400)) && (160 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2560))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2560) < 2400)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 2560) * 14) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 2560) / 160)) * 448) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 160)) + -6464)] : 0.000000e+00f);
  }
}

__kernel void fused_nn_max_pool2d_8_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 14; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 528; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 528) + ax3)] = -3.402823e+38f;
        for (int rv = 0; rv < 3; ++rv) {
          for (int rv1 = 0; rv1 < 3; ++rv1) {
            tensor[((((ax1 * 14) + ax2) * 528) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 528) + ax3)], (float)((((((1 - rv) <= ax1) && (ax1 < (15 - rv))) && ((1 - rv1) <= ax2)) && (ax2 < (15 - rv1))) ? placeholder[(((((((ax1 + rv) * 14) + ax2) + rv1) * 528) + ax3) + -7920)] : -3.402823e+38f));
          }
        }
      }
    }
  }
}


