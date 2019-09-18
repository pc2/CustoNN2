# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# tvm, relay
import tvm
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

from tensorflow.python.ops import variables


#import nnvm ir

import nnvm.frontend
import nnvm.compiler
import nnvm.symbol

# Base location for model related files.
repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'

# Test image
img_name = 'elephant-299.jpg'
image_url = os.path.join(repo_base, img_name)

######################################################################
# Tutorials
# ---------
# Please refer docs/frontend/tensorflow.md for more details for various models
# from tensorflow.

model_name = 'classify_image_graph_def-with_shapes.pb'
model_url = os.path.join(repo_base, model_name)

# Image label map
map_proto = 'imagenet_2012_challenge_label_map_proto.pbtxt'
map_proto_url = os.path.join(repo_base, map_proto)

# Human readable text for labels
label_map = 'imagenet_synset_to_human_label_map.txt'
label_map_url = os.path.join(repo_base, label_map)

# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
#target_host = 'llvm'
#layout = "NCHW"
#ctx = tvm.gpu(0)
target = 'aocl_sw_emu'
target_host = None
layout = None
ctx = tvm.cpu(0)

######################################################################
# Download required files
# -----------------------
# Download files listed above.
from tvm.contrib.download import download_testdata

img_path = download_testdata(image_url, img_name, module='data')
# model_path = download_testdata(model_url, model_name, module=['tf', 'InceptionV1'])
# map_proto_path = download_testdata(map_proto_url, map_proto, module='data')
# label_path = download_testdata(label_map_url, label_map, module='data')

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.
# model_path='/home/amey/.tvm_test_data/tf/InceptionV1/'

# model_path='/home/amey/Masters/project/tvm/build/models/inception_v1_inf_graph_t11.pb'
model_path='/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/tvm_models/inceptionv1/frozen.pb'
# model_path='/home/amey/Masters/project/tvm/build/models/frozen_graph.pb'
# model_path='/home/amey/.tvm_test_data/data/imagenet_2012_challenge_label_map_proto.pbtxt'

# with tf.gfile.GFile(model_path, 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     graph = tf.import_graph_def(graph_def, name='')
#     tf.train.write_graph(graph, 'models', 'train.pbtxt')


    # graph_def = tf_testing.ProcessGraphDefParam(graph_def)
 
# init_g = tf.global_variables_initializer()
# init_l = tf.local_variables_initializer()
with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf.Session() as sess:
        # sess.run(init_g)
        # sess.run(init_l)
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'InceptionV1/Logits/Predictions/Reshape_1')   

print("exited")


######################################################################
# Decode image
# ------------
# .. note::
#
#   tensorflow frontend import doesn't support preprocessing ops like JpegDecode.
#   JpegDecode is bypassed (just return source node).
#   Hence we supply decoded frame to TVM instead.
#

from PIL import Image
# image = Image.open(img_path).resize((299, 299))
image = Image.open(img_path).resize((224, 224))

x = np.array(image)


######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}
# tvm.relay.testing.tf.AddShapesToGraphDef

#either use nnvm or relay ir and comment the lines accordingly
###using relay ir
#sym, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
###end of relay ir

###using nnvm ir

sym, params = nnvm.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
###end of nnvm ir


print("Tensorflow protobuf imported to relay frontend.")
######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with tvm runtime.

#for nnvm ir
graph, lib, params = nnvm.compiler.build(sym,  target=target, target_host=target , params=params)

###for relay ir
'''
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(sym, target=target, target_host=target_host, params=params)
'''
