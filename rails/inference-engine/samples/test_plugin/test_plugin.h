// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to a folder with images or path to an image files: a .ubyte file for LeNet"\
                                    "and a .bmp file for the other networks.";

/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Path to a plugin folder.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. " \
                                            "Sample will look for a suitable plugin for device specified (CPU by default)";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance report";

/// @brief message for top results number
static const char ntop_message[] = "Number of top results (default 10)";

/// @brief message for iterations count
static const char iterations_count_message[] = "Number of iterations (default 1)";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for clDNN (GPU)-targeted custom kernels."\
                                            "Absolute path to the xml file with the kernels desc.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for MKLDNN (CPU)-targeted custom layers." \
                                                 "Absolute path to a shared library with the kernels impl.";

/// @brief message for plugin messages
static const char plugin_message[] = "Enables messages from a plugin";

// @brief message for CNN Model name argument.
static const char cnn_model[] = "Required. Input CNN Model name. Supported models : googlenet/resnet ";

// @brief message for Route XML argument.
static const char route_xml_message[] = "Required when the design=channel . Path to an .xml file with a routing configurations. ";

// @brief message for Labels file.
static const char label_message[] = "Path to the labels.txt file of the model with label indicies and names  ";

// @brief message for Labels file.
static const char bitstream_message[] = "Required. Path to the bitstreams directory ";

// @brief message for Labels file.
static const char design_message[] = " Type of CNN OpenCL architecture design. supported design : global/channel. Default : global";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Define parameter for set path to plugins <br>
DEFINE_string(pp, "", plugin_path_message);

/// @brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// @brief Top results number (default 10) <br>
DEFINE_int32(nt, 10, ntop_message);

/// @brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Iterations count (default 1)
DEFINE_int32(ni, 1, iterations_count_message);

/// @brief Enable plugin messages
DEFINE_bool(p_msg, false, plugin_message);

/// @brief Enable plugin messages
DEFINE_string(model, "", cnn_model);

/// @brief Absolute path to the Routing Configuration XML
DEFINE_string(route, "", route_xml_message);

/// @brief Absolute path to the Labels.txt file
DEFINE_string(label, "", label_message);

/// @brief Absolute path to the Directory of Bitstreams
DEFINE_string(bitstream, "", bitstream_message);

/// @brief Type of CNN OpenCL architecture design
DEFINE_string(design, "", design_message);


/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "Noctua Plugin's user application [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -nt \"<integer>\"         " << ntop_message << std::endl;
    std::cout << "    -model                  " << cnn_model << std::endl;
    std::cout << "    -route                  " << route_xml_message << std::endl;
    std::cout << "    -label                  " << label_message << std::endl;
    std::cout << "    -bitstream                  " << bitstream_message << std::endl;
    std::cout << "    -design                  " << design_message << std::endl;

}
