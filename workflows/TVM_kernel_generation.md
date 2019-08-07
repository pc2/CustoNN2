## Instructions for generating OpenCL kernels using TVM

#### Prerequisites:
- Tensorflow 1.13
- Python 3.6/3.5

#### Instructions:
- TVM needs a frozen protobuf `.pb` file for kernel generation.
- First, we download pre-trained `.ckpt` file from https://github.com/tensorflow/models/tree/master/research/slim
- We `.ckpt` file for generating the frozen `.pb` file.
- Then we clone https://github.com/tensorflow from git.
- We export the inference graph by using the following command inside the cloned directory 
`$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v1 \
  --output_file=/tmp/inception_v1_inf_graph.pb`

- We now convert the exported `.pb` to `.pbtxt` file since we need the last node from the exported model.
Conversion is done by using the following script https://gist.github.com/Arafatk/c063bddb9b8d17a037695d748db4f592.

- Within the same directory, we freeze the graph using `.ckpt` and `` files
`python freeze_graph.py
  --input_graph=inception_v1_inf_graph.pb
  --input_checkpoint=inception_v1.ckpt
  --input_binary=true
  --output_graph=frozen.pb
  --output_node_names=InceptionV1/Logits/Predictions/Reshape_1`

- In the above code `InceptionV1/Logits/Predictions/Reshape_1` is the last node in `.pb` file and `frozen.pb` is the exported frozen graph.

- This frozen file is provided to TVM for generating kernels.