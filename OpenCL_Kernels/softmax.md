TVM generated softmax takes in four inputs which are `tensor`, `input0`, `tensor1` and `tensor2` all of the array data structure with the type float.

In the softmax activation function, LOGITS scores are turned into probabilities. All probabilities sum up to one. providing probability distribution for the provided input scores. In TVM those LOGITS scores are provided by `input0` to the kernel.

on line 3 `tensor[0] = -3.402823e+38f;` kernel sets most negative number to compare all the input LOGITS with it.

```
for (int k1 = 0; k1 < 1001; ++k1) {
      tensor[0] = max(tensor[0], input0[k1]);
    }
```

Finds the max value for the provided LOGITS scores and stores them in `tensor` variable

`tensor1` is the temporary variable used to store the intermediate softmax values before their probabilities are distributed between `0-1`

```
for (int k2 = 0; k2 < 1001; ++k2) {
      tensor1[0] = (tensor1[0] + exp((input0[k2] - tensor[0])));
    }
```

These intermediate values are distributed between `0-1`
```
tensor2[ax1] = (exp((input0[ax1] - tensor[0])) / tensor1[0]);
```

`tensor2` is the output variable which is the output from the kernel.



