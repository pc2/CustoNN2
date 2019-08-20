__kernel void  Mul1__Fused_Mul__FusedScaleShift(__global float* restrict T_relu, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 150528; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_relu[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = ((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50176)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 50176)]);
    }
}


__kernel void  P_conv1_Conv2D(__global float* restrict T_transpose, __global float* restrict input0) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 158700; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_transpose[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((690 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52900)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52900) < 52210)) && (3 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 230))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 230) < 227)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52900) / 230) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 230)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52900)) + -2025)] : 0.000000e+00f);
    }
}

__kernel void  conv1_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[64];
#pragma unroll
    for( int j = 0; j < 64; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 64; ++ff) {
        //local memory for weights
        float input_weight[7*7*3];
#pragma unroll
        for( int k = 0; k < 7*7*3; ++k){
            input_weight[k] = input1[((ff * 7*7*3) + k)];
        }
        for (int yy = 0; yy < 112; ++yy) {
            for (int xx = 0; xx < 112; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;

                for (int rc = 0; rc < 3; ++rc) {
                    float temp_2 = 0.0;

                    for (int ry = 0; ry < 7; ++ry) {
                        float temp_1 = 0.0;

                        for (int rx = 0; rx < 7; ++rx) {
                            temp_1 += (input0[((((((((rc * 115) + yy) * 2) + ry) * 115) + xx) * 2) + rx)] * input_weight[(((((rc) * 7) + ry) * 7) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
               // temp_0 = (temp_0 > 0) ? temp_0 : 0.0;
                compute[((((ff * 112) + yy) * 112) + xx)] = temp_0;
            }
        }
    }
}


__kernel void  pool1_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
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



__kernel void  Mul1_1520_Fused_Mul__FusedScaleShift(__global float* restrict T_relu, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_relu[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]), 0.000000e+00f);
    }
}





__kernel void  block1_unit_1_bt_v2_shortcut_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[256];
#pragma unroll 64
    for( int j = 0; j < 256; ++j){
        input_bias[j] = input2[j];
    }
    
    for (int ff = 0; ff < 256; ++ff) {
        //local memory for weights
        float input_weight[64];
#pragma unroll
        for( int k = 0; k < 64; ++k){
            input_weight[k] = input1[((ff * 64) + k)];
        }
        
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 64; ++rc) {
                    temp_1 += (input0[((((rc * 56) + yy) * 56) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}


__kernel void  block1_unit_1_bt_v2_conv1_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[64];
#pragma unroll
    for( int j = 0; j < 64; j++){
        input_bias[j] = input2[j];
    }

    for (int ff = 0; ff < 64; ++ff) {
        //local memory for weights
        float input_weight[64];
#pragma unroll
        for( int k = 0; k < 64; k++){
            input_weight[k] = input1[((ff * 64) + k)];
        }
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 64; ++rc) {
                    temp_1 += (input0[((((rc * 56) + yy) * 56) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}


__kernel void P_block1_unit_1_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f);
    }
}


__kernel void  block1_unit_1_bt_v2_conv2_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[64];
#pragma unroll
    for( int j = 0; j < 64; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 64; ++ff) {
        //local memory for weights
        float input_weight[3*3*64];
#pragma unroll
        for( int k = 0; k < 3*3*64; ++k){
            input_weight[k] = input1[((ff * 3*3*64) + k)];
        }
        
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 64; ++rc) {
                    float temp_2 = 0.0;
                    for (int ry = 0; ry < 3; ++ry) {
                        float temp_1 = 0.0;
                        for (int rx = 0; rx < 3; ++rx) {
                            temp_1 += (input0[((((((rc * 58) + yy) + ry) * 58) + xx) + rx)] * input_weight[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}



__kernel void  block1_unit_1_bt_v2_conv3_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[256];
#pragma unroll 64
    for( int j = 0; j < 256; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 256; ++ff) {
        //local memory for weights
        float input_weight[64];
#pragma unroll
        for( int k = 0; k < 64; ++k){
            input_weight[k] = input1[((ff * 64) + k)];
        }
        
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 64; ++rc) {
                    temp_1 += (input0[((((rc * 56) + yy) * 56) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}



__kernel void  block1_unit_1_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] =  input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
    }
}




__kernel void Mul1_1547_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]), 0.000000e+00f);
    }
}


__kernel void  block1_unit_2_bt_v2_conv1_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[64];
#pragma unroll
    for( int j = 0; j < 64;++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 64; ++ff) {
        //local memory for weights
        float input_weight[256];
#pragma unroll 64
        for( int k = 0; k < 256; ++k){
            input_weight[k] = input1[((ff * 256) + k)];
        }
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 256; ++rc) {
                    temp_1 += (input0[((((rc * 56) + yy) * 56) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}


__kernel void P_block1_unit_2_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f);
    }
    
}

__kernel void  block1_unit_2_bt_v2_conv2_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[64];
#pragma unroll
    for( int j = 0; j < 64; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 64; ++ff) {
        //local memory for weights
        float input_weight[3*3*64];
#pragma unroll
        for( int k = 0; k < 3*3*64; ++k){
            input_weight[k] = input1[((ff * 3*3*64) + k)];
        }
        
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 64; ++rc) {
                    float temp_2 = 0.0;
                    for (int ry = 0; ry < 3; ++ry) {
                        float temp_1 = 0.0;
                        for (int rx = 0; rx < 3; ++rx) {
                            temp_1 += (input0[((((((rc * 58) + yy) + ry) * 58) + xx) + rx)] * input_weight[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}


__kernel void  block1_unit_2_bt_v2_conv3_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[256];
#pragma unroll 64
    for( int j = 0; j < 256; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 256; ++ff) {
        //local memory for weights
        float input_weight[64];
#pragma unroll
        for( int k = 0; k < 64; ++k){
            input_weight[k] = input1[((ff * 64) + k)];
        }
        
        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input2[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 64; ++rc) {
                    temp_1 += (input0[((((rc * 56) + yy) * 56) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}



__kernel void block1_unit_2_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
    }
}


__kernel void  block1_unit_3_bt_v2_shortcut_MaxPool(__global float* restrict tensor, __global float* restrict input0) {
    for (int ax1 = 0; ax1 < 256; ++ax1) {
        for (int ax2 = 0; ax2 < 28; ++ax2) {
            for (int ax3 = 0; ax3 < 28; ++ax3) {
                tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
                tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], input0[(((((ax1 * 28) + ax2) * 56) + ax3) * 2)]);
            }
        }
    }
}



__kernel void Mul1_1574_Fused_Mul__FusedScaleShift(__global float* restrict T_pad, __global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 802816; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = max(((input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner] * input1[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]) + input2[(ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3136)]), 0.000000e+00f);
    }
}



__kernel void  block1_unit_3_bt_v2_conv1_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases

    __local float input_bias[64];
    #pragma unroll
    for( int j = 0; j < 64; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 64; ++ff) {
        //local memory for weights
        float input_weight[256];
#pragma unroll 64
        for( int k = 0; k < 256; ++k){
            input_weight[k] = input1[((ff * 256) + k)];
        }

        for (int yy = 0; yy < 56; ++yy) {
            for (int xx = 0; xx < 56; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 256; ++rc) {
                    temp_1 += (input0[((((rc * 56) + yy) * 56) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                compute[((((ff * 56) + yy) * 56) + xx)] = temp_0;
            }
        }
    }
}


__kernel void P_block1_unit_3_bt_v2_conv2_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f);
    }
    
}

__kernel void  block1_unit_3_bt_v2_conv2_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[64];
#pragma unroll
    for( int j = 0; j < 64; ++j){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 64; ++ff) {
        //local memory for weights
        float input_weight[3*3*64];
#pragma unroll
        for( int k = 0; k < 3*3*64; ++k){
            input_weight[k] = input1[((ff * 3*3*64) + k)];
        }
        
        for (int yy = 0; yy < 28; ++yy) {
            for (int xx = 0; xx < 28; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_3 = 0.0;
                for (int rc = 0; rc < 64; ++rc) {
                    float temp_2 = 0.0;
                    for (int ry = 0; ry < 3; ++ry) {
                        float temp_1 = 0.0;
                        for (int rx = 0; rx < 3; ++rx) {
                            temp_1 += (input0[((((((((rc * 29) + yy) * 2) + ry) * 29) + xx) * 2) + rx)] * input_weight[(((((rc) * 3) + ry) * 3) + rx)]);
                        }
                        temp_2 += temp_1;
                    }
                    temp_3 += temp_2;
                }
                temp_0 += temp_3;
                temp_0 = (temp_0 > 0) ? temp_0 : 0.000000e+00f;
                compute[((((ff * 28) + yy) * 28) + xx)] = temp_0;
            }
        }
    }
}



__kernel void  block1_unit_3_bt_v2_conv3_Conv2D(__global float* restrict compute,__global float* restrict input0, __global float* restrict input1, __global float* restrict input2) {
    //local memory for biases
    __local float input_bias[256];
#pragma unroll 64
    for( int j = 0; j < 256; j++){
        input_bias[j] = input2[j];
    }
    for (int ff = 0; ff < 256; ++ff) {
        //local memory for weights
        float input_weight[64];
#pragma unroll 64
        for( int k = 0; k < 64; k++){
            input_weight[k] = input1[((ff * 64) + k)];
        }
        
        for (int yy = 0; yy < 28; ++yy) {
            for (int xx = 0; xx < 28; ++xx) {
                float temp_0 = input_bias[ff];
                float temp_1 = 0.0;
                for (int rc = 0; rc < 64; ++rc) {
                    temp_1 += (input0[((((rc * 28) + yy) * 28) + xx)] * input_weight[(rc)]);
                }
                temp_0 += temp_1;
                compute[((((ff * 28) + yy) * 28) + xx)] = temp_0;
            }
        }
    }
}




__kernel void  block1_unit_3_bt_v2_add(__global float* restrict T_add, __global float* restrict input0, __global float* restrict input1) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 200704; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
        T_add[ax0_ax1_fused_ax2_fused_ax3_fused_inner] =  input0[ax0_ax1_fused_ax2_fused_ax3_fused_inner ] + input1[ax0_ax1_fused_ax2_fused_ax3_fused_inner];
    }
}

