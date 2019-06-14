__kernel void Logits_Predictions_Softmax(__global float* restrict input0, 
					__global float* restrict tensor2) {
  float tensor,tensor1;
  for (int ax1 = 0; ax1 < 1001; ++ax1) {
    tensor = -3.402823e+38f;
    for (int k1 = 0; k1 < 1001; ++k1) {
      tensor = max(tensor, input0[k1]);
    }
    tensor1[0] = 0.000000e+00f;
    for (int k2 = 0; k2 < 1001; ++k2) {
      tensor1 = (tensor1 + exp((input0[k2] - tensor)));
    }
    tensor2[ax1] = (exp((input0[ax1] - tensor)) / tensor1);
  }
}
