## Tensorflow execution of Googlenet 
- Tensorflow implementation (a python file) in this dir allows us to run any frozen pb file
- Make sure you have Tensorflow , scipy , opencv , numpy packages. 
- We can take advantage of this to run Googlent for verification of our kernel implementations
- To run the implementation until any given layer , just change `l_output = graph.get_tensor_by_name('<insert layer name>')` for example `l_output = graph.get_tensor_by_name('InceptionV1/InceptionV1/MaxPool_3a_3x3/MaxPool:0')` will run until layer 3a Maxpool.
   Please refer to [TF Googlenet tensor names](https://git.uni-paderborn.de/cs-hit/pg-custonn2-2018/blob/tvm/OpenCL_Kernels/Testing_code/TF_Googlenet_Layers.txt) 
- The output will be stored in  the variable `Session_out`
- You can print out the contents of `Session_out` for any layer to see the output.
- The format used by TF is NHWC in this case. So modify your print statements appropriatley.
- TODO: The results of all the layers are to be stored in a common directory