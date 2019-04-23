#include <string>
#include <vector>
#include <iostream>

#include "CL/cl.hpp"
#include "fpga_plugin.h"
#include <format_reader_ptr.h>

#include <inference_engine.hpp>

using namespace InferenceEngine;

//Data structure to store information of each layer
struct layersDetails
{
  std::string layerName;
  std::string layerType;
  std::vector<float> layerBias;
  std::vector<float> layerWeights;
  std::map<std::string,std::string> params;
};

unsigned char *images;

void parse_images(std::vector<std::string> imageNames, unsigned char *images, InferenceEngine::CNNNetwork network)
{

  InputsDataMap inputInfo = network.getInputsInfo();
  if (inputInfo.size() != 1)
    throw std::logic_error("Sample supports topologies only with 1 input");
  auto inputInfoItem = *inputInfo.begin();

  // Specifying the precision and layout of input data provided by the user.
  // This should be called before load of the network to the plugin
  inputInfoItem.second->setPrecision(Precision::U8);
  inputInfoItem.second->setLayout(Layout::NCHW);

  std::vector<std::shared_ptr<unsigned char>> imagesData;
  for (auto &i : imageNames)
  {
    FormatReader::ReaderPtr reader(i.c_str());
    if (reader.get() == nullptr)
    {
      std::cout << "Image " + i + " cannot be read!" << std::endl;
      continue;
    }
    // Store image data
    std::shared_ptr<unsigned char> data(
        reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3],
                        inputInfoItem.second->getTensorDesc().getDims()[2]));
    if (data.get() != nullptr)
    {
      imagesData.push_back(data);
    }
  }
  if (imagesData.empty())
    throw std::logic_error("Valid input images were not found!");

  /* TODO: imagesData[i] : ERROR at Line 62
  
  images = new unsigned char[inputInfoItem.second->getTensorDesc().getDims()[3]*inputInfoItem.second->getTensorDesc().getDims()[2]*imagesData.size()];
	int img_index = 0;
	for(int i=0;i<imagesData.size();i++)
	{
		for(int j=0;j<inputInfoItem.second->getTensorDesc().getDims()[3]*inputInfoItem.second->getTensorDesc().getDims()[2];j++)
		{
			images[img_index] = imagesData.at(i)[j];
			img_index++;
		}
	} */
}

void print_layerDetails(std::vector<layersDetails> cnnlayers)
{

  for (layersDetails a : cnnlayers)
  {
    std::cout << " LayerName: " << a.layerName<<std::endl;
    std::cout << " LayerType: " << a.layerType<<std::endl;
    std::cout << "Bias:";
    for (int i = 0; i < a.layerBias.size(); i++)
    {
      std::cout << a.layerBias[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Weights:";
    for (int i = 0; i < a.layerWeights.size(); i++)
    {
      //std::cout << a.layerWeights[i] << " ";
    }
    for(auto const &y : a.params){
      std::cout<<y.first << ":"<<y.second<<std::endl;
    }
  }
  std::cout << std::endl;
}

std::string bitstreamFinder(char *filepath)
{

  char *full_filename;
  char *filenameFromPath;

  strtok(filepath, "/");
  while ((filenameFromPath = strtok(NULL, "/")) != NULL)
  {
    full_filename = filenameFromPath;
  }
  std::string str = "";
  str = full_filename;
  size_t lastindex = str.find_last_of(".");
  std::string filename = str.substr(0, lastindex);
  filename += ".aocx";
  std::string str1 = "kernels/" + filename;
  char char1[20];
  strcpy(char1, str1.c_str());
  FILE *fp = fopen(char1, "r");
  if (fp != NULL)
  {
    return filename;
  }
  else
  {
    return "not found";
  }
}

void fpga_launcher(InferenceEngine::CNNNetwork network, char *model_path, std::vector<std::string> imageNames)
{
  std::cout<<"In FPGA Launcher"<<std::endl;
  std::string overlay_name = bitstreamFinder(model_path); //Checking the availability of bitstream
  if(overlay_name=="not found"){
    std::cout<<" Bitstream not found\n";
    //exit(0);  
  }
  parse_images(imageNames, images, network);

  cl_int err;

  std::vector<cl::Platform> PlatformList; //Platforms

  err = cl::Platform::get(&PlatformList);
  assert(err == CL_SUCCESS);

   for (int i=0; i<PlatformList.size(); i++)
        {
                printf("Platform Number: %d\n", i);
                std::cout << "Platform Name: "<<PlatformList.at(i).getInfo<CL_PLATFORM_NAME>()<<"\n";
                std::cout << "Platform Profile: "<<PlatformList.at(i).getInfo<CL_PLATFORM_PROFILE>()<<"\n";
                std::cout << "Platform Version: "<<PlatformList.at(i).getInfo<CL_PLATFORM_VERSION>()<<"\n";
                std::cout << "Platform Vendor: "<<PlatformList.at(i).getInfo<CL_PLATFORM_VENDOR>()<<"\n\n";
        }



  std::vector<cl::Device> DeviceList; //Devices

  err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList);
  assert(err == CL_SUCCESS);

   for (int i=0; i<DeviceList.size(); i++)
        {
                printf("Device Number: %d\n", i);
                std::cout << "Device Name: "<<DeviceList.at(i).getInfo<CL_DEVICE_NAME>()<<"\n";
                std::cout << "Device Vendor: "<<DeviceList.at(i).getInfo<CL_DEVICE_VENDOR>()<<"\n";
                std::cout << "Is Device Available?: "<<DeviceList.at(i).getInfo<CL_DEVICE_AVAILABLE>()<<"\n";
                std::cout << "Is Device Little Endian?: "<<DeviceList.at(i).getInfo<CL_DEVICE_ENDIAN_LITTLE>()<<"\n";
                std::cout << "Device Max Compute Units: "<<DeviceList.at(i).getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()<<"\n";
                std::cout << "Device Max Work Item Dimensions: "<<DeviceList.at(i).getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()<<"\n";
                std::cout << "Device Max Work Group Size: "<<DeviceList.at(i).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()<<"\n";
                std::cout << "Device Max Frequency: "<<DeviceList.at(i).getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()<<"\n";
                std::cout << "Device Max Mem Alloc Size: "<<DeviceList.at(i).getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()<<"\n\n";
        }


  cl::Context mycontext(DeviceList); //Context
  assert(err == CL_SUCCESS);

  details::CNNNetworkIterator it(network.actual);
  int no_of_layers = static_cast<int>(network.layerCount());

  //Get the Layers in the network and store the weights and bias in a structure for each layer.

  std::vector<layersDetails> cnnLayersList;
  while (it != details::CNNNetworkIterator())
  {
    CNNLayer::Ptr layer = *it;
    layersDetails layerinfo;
    layerinfo.layerName = layer->name;
    layerinfo.layerType = layer->type;
    layerinfo.params = layer->params;
    std::cout<<"Parsing Kernel:" <<layer->name<<std::endl;

    //Parameters print:
    //std::cout<<"Parameters for the layer are:"<<std::endl;
    //for(auto const &y : layer->params){
      //std::cout<<y.first << ":"<<y.second<<std::endl;
    //}

    //store the bias and weights
    for (auto const &x : layer->blobs)
    {
      if (x.first == "biases")
      {
        for (int biasCount = 0; biasCount < x.second->size(); biasCount++)
        {
          layerinfo.layerBias.push_back(x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[biasCount]); // put the layer bias in the list
        }
      }
      else if (x.first == "weights")
      {
        for (int m = 0; m < x.second->size(); m++)
        {
          layerinfo.layerWeights.push_back(x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[m]);
        }
      }
    }
    cnnLayersList.push_back(layerinfo); // add the layer information to the List of structure.
    it++;
  }

  print_layerDetails(cnnLayersList);

  cl::CommandQueue myqueue(mycontext, DeviceList[0]); 	//command queue
  assert(err==CL_SUCCESS);
/*
  cl::CommandQueue *queues[50];
  cl::Buffer *buffers[100];
  for (int i = 0; i < no_of_layers; i++)
  {
    CNNLayer::Ptr layer = *it;
    queues[i] = new cl::CommandQueue(mycontext, DeviceList[0]);
  } */
 
 
/*  
// Launching of Kernel
	
const char *conv_kernel_name = "ConvolutionLayer";
const char *max_kernel_name = "MaxPool";
const char *fcl_kernel_name = "FCL_Kernel";
std::ifstream aocx_stream("SimpleCNN.aocx", std::ios::in|std::ios::binary);
checkErr(aocx_stream.is_open() ? CL_SUCCESS:-1, "SimpleCNN.aocx");
std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));

cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()+1));	
cl::Program program(mycontext,DeviceList[0],mybinaries);  

//TODO Buffer for Inputs and outputs
//TODO Writing data to the device

*/  
  
//creating  kernel
/*
cl::Kernel Convkernel(program,conv_kernel_name);
assert(err==CL_SUCCESS);
cl::Kernel Maxkernel(program,max_kernel_name);
assert(err==CL_SUCCESS);
cl::Kernel FCLkernel(program,fcl_kernel_name);
assert(err==CL_SUCCESS);


layersDetails conv1;
for(auto const& value: cnnLayersList) {
	if (value.layerName == "conv1") {
		conv1 = value;
	}
}
err = Convkernel.setArg(0, images);
assert(err==CL_SUCCESS);
err = Convkernel.setArg(1, conv1_output);
assert(err==CL_SUCCESS);
err = Convkernel.setArg(2, conv1.layerBias);
assert(err==CL_SUCCESS);
err = Convkernel.setArg(3, conv1.layerWeights);
assert(err==CL_SUCCESS);
err = Convkernel.setArg(4, conv1.number_of_filters);
assert(err==CL_SUCCESS);
err = Convkernel.setArg(5, conv1.number_of_image_rows);
assert(err==CL_SUCCESS);
err = Convkernel.setArg(6, conv1.number_of_image_cols);
assert(err==CL_SUCCESS);
err = Convkernel.setArg(7, conv1.conv_stride);
assert(err==CL_SUCCESS);


layersDetails pool1;
for(auto const& value: cnnLayersList) {
	if (value.layerName == "pool1") {
		pool1 = value;
	}
}
err = Maxkernel.setArg(0, conv1_output);
assert(err==CL_SUCCESS);
err = Maxkernel.setArg(1, pool1_output);
assert(err==CL_SUCCESS);
err = Maxkernel.setArg(2, pool1.number_of_filters);
assert(err==CL_SUCCESS);
err = Maxkernel.setArg(3, pool1.number_of_image_rows);
assert(err==CL_SUCCESS);
err = Maxkernel.setArg(4, pool1.number_of_image_cols);
assert(err==CL_SUCCESS);
	

layersDetails ip1;
for(auto const& value: cnnLayersList) {
	if (value.layerName == "ip1") {
		ip1 = value;
	}
}
err = FCLkernel.setArg(0, pool1_output);
assert(err==CL_SUCCESS);
err = FCLkernel.setArg(1, ip1.layerWeights);
assert(err==CL_SUCCESS);
err = FCLkernel.setArg(2, output_labels);
assert(err==CL_SUCCESS);
err = FCLkernel.setArg(3, ip1.number_of_filters);
assert(err==CL_SUCCESS);
err = FCLkernel.setArg(4, 10);
assert(err==CL_SUCCESS);
*/
}
