## Intel OpenVINO Model Optimizer instructions:-

OpenVINO supports a number of topologies trained using different frameworks such as Tensorflow, Caffe etc. This document provides instructions on converting a pre-trained tensorflow model from the Slim Image classification library.

Steps:-

- Download one of the supported topologies from the OpenVINO website https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html
- For a Tensorflow model, a .ckpt (checkpoint) file is obtained. In order to generate IR, Model Optimizer needs an inference Graph file, i.e. a protobuf file (.pb).
- To obtain a .pb file, run the export_inference_graph.py script which is available as a part of the Slim repository, `python3 tf_models/research/slim/export_inference_graph.py \
    --model_name inception_v1 \
    --output_file inception_v1_inference_graph.pb`
- To convert to IR, run `<MODEL_OPTIMIZER_INSTALL_DIR>/mo_tf.py --input_model ./inception_v1_inference_graph.pb --input_checkpoint ./inception_v1.ckpt -b 1 --mean_value [127.5,127.5,127.5] --scale 127.5`
- In order to obtain additional information about the inference graph of the selected topology another python script called summarize_graph.py is available as a part of the Model Optimizer. `python3 <MODEL_OPTIMIZER_INSTALL_DIR>/mo/utils/summarize_graph.py --input_model ./inception_v1_inference_graph.pb`