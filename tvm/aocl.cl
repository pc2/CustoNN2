__kernel void myadd_kernel0(__global float* restrict C, __global float* restrict A, __global float* restrict B, int n) {
  for (int i_inner = 0; i_inner < n; ++i_inner) {
    C[i_inner] = (A[i_inner] / B[i_inner]);
  }
}

