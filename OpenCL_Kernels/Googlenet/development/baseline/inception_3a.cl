__kernel void Padding_Conv2d_1a_7x7_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 157323; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((458 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) < 51754)) && (2 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229) < 226)) ? input0[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 52441) / 229) * 224) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 229)) * 3) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner / 52441)) + -1350)] : 0.000000e+00f);
    }
}

__kernel void Conv2d_1a_7x7_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)


    

{
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 112; ++yy)
        {
            for (int xx = 0; xx < 112; ++xx)
            {
                compute[((((ff * 112) + yy) * 112) + xx)] = input2[ff];
                for (int rc = 0; rc < 3; ++rc)
                {
                    for (int ry = 0; ry < 7; ++ry)
                    {
                        for (int rx = 0; rx < 7; ++rx)
                        {
                            compute[((((ff * 112) + yy) * 112) + xx)] = (compute[((((ff * 112) + yy) * 112) + xx)] + (input0[(((((rc * 52441) + (yy * 458)) + (ry * 229)) + (xx * 2)) + rx)] * input1[((((((ff * 3) + rc) * 7) + ry) * 7) + rx)]));
                        }
                    }
                }
                compute[((((ff * 112) + yy) * 112) + xx)] = (compute[((((ff * 112) + yy) * 112) + xx)] > 0) ? compute[((((ff * 112) + yy) * 112) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void MaxPool_2a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 64; ++ax1)
    {
        for (int ax2 = 0; ax2 < 56; ++ax2)
        {
            for (int ax3 = 0; ax3 < 56; ++ax3)
            {
                tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 56) + ax2) * 56) + ax3)] = max(tensor[((((ax1 * 56) + ax2) * 56) + ax3)], (float)((((ax2 * 2) < (112 - rv)) && ((ax3 * 2) < (112 - rv1))) ? input0[((((((((ax1 * 56) + ax2) * 2) + rv) * 56) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}

__kernel void Conv2d_2b_1x1_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 64; ++ff)
    {
        for (int yy = 0; yy < 56; ++yy)
        {
            for (int xx = 0; xx < 56; ++xx)
            {
                compute[((((ff * 56) + yy) * 56) + xx)] = input2[ff];
                for (int rc = 0; rc < 64; ++rc)
                {
                    compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (input0[((((rc * 56) + yy) * 56) + xx)] * input1[((ff * 64) + rc)]));
                }
                compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] > 0) ? compute[((((ff * 56) + yy) * 56) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void Padding_Conv2d_2c_3x3_Conv2D(__global float *restrict T_pad, __global float *restrict input0)
{
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 215296; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner)
    {
        T_pad[ax0_ax1_fused_ax2_fused_ax3_fused_inner] = (float)(((((58 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364)) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) < 3306)) && (1 <= (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58))) && ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58) < 57)) ? input0[((((((ax0_ax1_fused_ax2_fused_ax3_fused_inner / 3364) * 56) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner % 3364) / 58)) * 56) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner % 58)) + -57)] : 0.000000e+00f);
    }
}

__kernel void Conv2d_2c_3x3_Conv2D(__global float *restrict compute, __global float *restrict input0, __global float *restrict input1, __global float *restrict input2)
{
    for (int ff = 0; ff < 192; ++ff)
    {
        for (int yy = 0; yy < 56; ++yy)
        {
            for (int xx = 0; xx < 56; ++xx)
            {
                compute[((((ff * 56) + yy) * 56) + xx)] = input2[ff];
                for (int rc = 0; rc < 64; ++rc)
                {
                    for (int ry = 0; ry < 3; ++ry)
                    {
                        for (int rx = 0; rx < 3; ++rx)
                        {
                            compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] + (input0[((((((rc * 58) + yy) + ry) * 58) + xx) + rx)] * input1[((((((ff * 64) + rc) * 3) + ry) * 3) + rx)]));
                        }
                    }
                }
                compute[((((ff * 56) + yy) * 56) + xx)] = (compute[((((ff * 56) + yy) * 56) + xx)] > 0) ? compute[((((ff * 56) + yy) * 56) + xx)] : 0.000000e+00f;
            }
        }
    }
}

__kernel void MaxPool_3a_3x3_MaxPool(__global float *restrict tensor, __global float *restrict input0)
{
    for (int ax1 = 0; ax1 < 192; ++ax1)
    {
        for (int ax2 = 0; ax2 < 28; ++ax2)
        {
            for (int ax3 = 0; ax3 < 28; ++ax3)
            {
                tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = -3.402823e+38f;
                for (int rv = 0; rv < 3; ++rv)
                {
                    for (int rv1 = 0; rv1 < 3; ++rv1)
                    {
                        tensor[((((ax1 * 28) + ax2) * 28) + ax3)] = max(tensor[((((ax1 * 28) + ax2) * 28) + ax3)], (float)((((ax2 * 2) < (56 - rv)) && ((ax3 * 2) < (56 - rv1))) ? input0[((((((((ax1 * 28) + ax2) * 2) + rv) * 28) + ax3) * 2) + rv1)] : -3.402823e+38f));
                    }
                }
            }
        }
    }
}