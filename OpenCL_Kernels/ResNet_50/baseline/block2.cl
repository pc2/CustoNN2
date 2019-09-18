__kernel void Mul1_1601_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f);
 }
}

__kernel void  block2_unit_1_bt_v2_shortcut_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
        }
      }
    }
  }
}


__kernel void  block2_unit_1_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 256; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 256) + rc)]));
        }
	compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
      }
    }
  }
}


__kernel void P_block2_unit_1_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
  }
}

__kernel void  block2_unit_1_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 128; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
	compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
      }
    }
  }
}



__kernel void  block2_unit_1_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 128; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 128) + rc)]));
        }
      }
    }
  }
}


__kernel void  block2_unit_1_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}



__kernel void Mul1_1628_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f);
 }
}



__kernel void  block2_unit_2_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 512) + rc)]));
        }
	compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;      
      }
    }
  }
}


__kernel void P_block2_unit_2_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
  }
}


__kernel void  block2_unit_2_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 128; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
	compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
      }
    }
  }
}


__kernel void  block2_unit_2_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 128; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 128) + rc)]));
        }
      }
    }
  }
}


__kernel void block2_unit_2_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner ] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}



__kernel void Mul1_1655_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f);
 }
}



__kernel void  block2_unit_3_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 512) + rc)]));
        }
	compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
      }
    }
  }
}


__kernel void P_block2_unit_3_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
  }
}


__kernel void  block2_unit_3_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 128; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((((rc * 30) + yy) + ry) * 30) + xx) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
	compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
      }
    }
  }
}


__kernel void  block2_unit_3_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 128; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 128) + rc)]));
        }
      }
    }
  }
}


__kernel void  block2_unit_3_bt_v2_add(__global float* restrict T_add,  __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}


__kernel void  block2_unit_4_bt_v2_shortcut_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
  for (int ax1 = 0; ax1 < 512; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = -3.402823e+38f;
        tensor[((((ax1 * 14) + ax2) * 14) + ax3)] = max(tensor[((((ax1 * 14) + ax2) * 14) + ax3)], input0[(((((ax1 * 14) + ax2) * 28) + ax3) * 2)]);
      }
    }
  }
}




__kernel void Mul1_1682_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 401408; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 784)]), 0.000000e+00f);
 }
}

__kernel void  block2_unit_4_bt_v2_conv1_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((((ff * 28) + yy) * 28) + xx)] = input2[ff];
        for (int rc = 0; rc < 512; ++rc) {
          compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] + (input0[((((rc * 28) + yy) * 28) + xx)] * input1[((ff * 512) + rc)]));
        }
	compute[((((ff * 28) + yy) * 28) + xx)] = (compute[((((ff * 28) + yy) * 28) + xx)] > 0) ? compute[((((ff * 28) + yy) * 28) + xx)] : 0.000000e+00f;
      }
    }
  }
}


__kernel void P_block2_unit_4_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 115200; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((30 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) < 870)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30) < 29)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 900) * 28) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 900) / 30)) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 30)) + -29)] : 0.000000e+00f);
  }
}


__kernel void  block2_unit_4_bt_v2_conv2_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 128; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 128; ++rc) {
          for (int ry = 0; ry < 3; ++ry) {
            for (int rx = 0; rx < 3; ++rx) {
              compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((((((rc * 15) + yy) * 2) + ry) * 15) + xx) * 2) + rx)] * input1[((((((ff * 128) + rc) * 3) + ry) * 3) + rx)]));
            }
          }
        }
	compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] > 0) ? compute[((((ff * 14) + yy) * 14) + xx)] : 0.000000e+00f;
      }
    }
  }
}



__kernel void  block2_unit_4_bt_v2_conv3_Conv2D(__global float* restrict compute, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
  for (int ff = 0; ff < 512; ++ff) {
    for (int yy = 0; yy < 14; ++yy) {
      for (int xx = 0; xx < 14; ++xx) {
        compute[((((ff * 14) + yy) * 14) + xx)] = input2[ff];
        for (int rc = 0; rc < 128; ++rc) {
          compute[((((ff * 14) + yy) * 14) + xx)] = (compute[((((ff * 14) + yy) * 14) + xx)] + (input0[((((rc * 14) + yy) * 14) + xx)] * input1[((ff * 128) + rc)]));
        }
      }
    }
  }
}



__kernel void block2_unit_4_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 100352; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner ] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
  }
}
