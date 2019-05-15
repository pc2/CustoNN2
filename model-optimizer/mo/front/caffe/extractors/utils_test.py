"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest
from unittest.mock import patch, call

import numpy as np

from mo.front.caffe.extractors.utils import weights_biases, embed_input, get_canonical_axis_index
from mo.utils.unittest.extractors import FakeModelLayer


class TestWeightsBiases(unittest.TestCase):
    def test_weights_biases_no_layer_no_bias(self):
        res = weights_biases(False, None)
        self.assertEqual(res, {})

    @patch('mo.front.caffe.extractors.utils.embed_input')
    def test_weights_biases_layer_no_bias(self, embed_input_mock):
        weights_biases(False, FakeModelLayer([[1, 2], ]))
        calls = [call({}, 1, 'weights', [1, 2])]
        embed_input_mock.assert_has_calls(calls)

    @patch('mo.front.caffe.extractors.utils.embed_input')
    def test_weights_biases_layer_bias(self, embed_input_mock):
        weights_biases(True, FakeModelLayer([[1, 2], [3, 4]]))
        calls = [call({}, 1, 'weights', [1, 2]), call({}, 2, 'biases', [3, 4])]
        embed_input_mock.assert_has_calls(calls)


class TestEmbedInput(unittest.TestCase):
    def test_embed_input_no_bin_name_no_bias(self):
        attrs = {}
        blob = np.array([1, 2])
        name = 'weights'
        embed_input(attrs, 1, name, blob, None)
        exp_res = {
            'weights': blob,
            'embedded_inputs': [
                (1, name, {'bin': name})
            ]
        }
        for key in exp_res.keys():
            if key == name:
                np.testing.assert_equal(attrs[key], exp_res[key])
            else:
                self.assertEqual(attrs[key], exp_res[key])

    def test_embed_input_w_bin_name(self):
        attrs = {}
        blob = np.array([1, 2])
        name = 'weights'
        embed_input(attrs, 1, name, blob, 'special_name')
        exp_res = {
            'weights': blob,
            'embedded_inputs': [
                (1, name, {'bin': 'special_name'})
            ]
        }
        for key in exp_res.keys():
            if key == name:
                np.testing.assert_equal(attrs[key], exp_res[key])
            else:
                self.assertEqual(attrs[key], exp_res[key])


class TestCanonicalAxisIndex(unittest.TestCase):
    def test_negative_index(self):
        shape = [1, 2, 3, 4]
        inds = [-4, -3, -2, -1]
        expected_inds = [0, 1, 2, 3]
        for i in range(len(inds)):
            assert get_canonical_axis_index(shape, inds[i]) == expected_inds[i]

    def test_posirive_index(self):
        shape = [1, 2, 3, 4]
        inds = [0, 1, 2, 3]
        expected_inds = [0, 1, 2, 3]
        for i in range(len(inds)):
            assert get_canonical_axis_index(shape, inds[i]) == expected_inds[i]
