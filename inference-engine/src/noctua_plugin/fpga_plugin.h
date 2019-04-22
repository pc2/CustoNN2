#include "CL/cl.hpp"
#include <inference_engine.hpp>
#include <string>
#include <iostream>

using namespace InferenceEngine;

void fpga_launcher(InferenceEngine::CNNNetwork network, char *model_path);
std::string bitstreamFinder(char *model_path);
void parse_images(std::vector<std::string> imageNames,unsigned char *images,InferenceEngine::CNNNetwork network);
