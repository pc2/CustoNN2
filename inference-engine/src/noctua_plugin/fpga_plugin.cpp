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
  float *layerBias;
  float *layerWeights;
  int num_biases;
  int num_weights;
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
      //unsigned char test=data.get()[0];
      //std::cout<<unsigned(test)<<std::endl;
    }
  }
  if (imagesData.empty())
    throw std::logic_error("Valid input images were not found!");
  //Total number of images * dimension
  images = new unsigned char[inputInfoItem.second->getTensorDesc().getDims()[3]*inputInfoItem.second->getTensorDesc().getDims()[2]*imagesData.size()];
  int numberOfPixels=inputInfoItem.second->getTensorDesc().getDims()[3]*inputInfoItem.second->getTensorDesc().getDims()[2];
	int img_index = 0;
  std::cout<<"Number of Pixels:"<<numberOfPixels<<" , Number of Images:"<<imagesData.size() <<std::endl;
	for(int i=0;i<imagesData.size();i++)
	{
		for(int j=0;j<numberOfPixels;j++)
		{
			images[i*numberOfPixels + j] = imagesData.at(i).get()[j];
      //std::cout<<unsigned(imagesData.at(i).get()[j])<<" ";
			img_index++;
		}
	}
  std::cout<<"Images Pixel: " <<std::endl;
  for(int k=0;k<numberOfPixels;k++){
    std::cout<<k <<":"<< unsigned(images[k]) <<std::endl;
  }
  std::cout<<"Number of Pixels:"<<numberOfPixels<<" , Number of Images:"<<imagesData.size() <<std::endl;
}

void print_layerDetails(std::vector<layersDetails> cnnlayers)
{

  for (layersDetails a : cnnlayers)
  {
    std::cout << " LayerName: " << a.layerName<<std::endl;
    std::cout << " LayerType: " << a.layerType<<std::endl;
    std::cout << "Bias:";
    for (int i = 0; i < a.num_biases; i++)
    {
      std::cout << a.layerBias[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Weights:";
    for (int i = 0; i < a.num_weights; i++)
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

int fpga_launcher(InferenceEngine::CNNNetwork network, char *model_path, std::vector<std::string> imageNames)
{
  std::cout<<"In FPGA Launcher"<<std::endl;
  std::string overlay_name = bitstreamFinder(model_path); //Checking the availability of bitstream
  if(overlay_name=="not found"){
    std::cout<<" Bitstream not found\n";
   //return -1; 
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
    layerinfo.num_biases = 0;
    layerinfo.num_weights = 0;
    std::cout<<"Parsing Kernel:" <<layer->name<<std::endl;

    //store the bias and weights
    for (auto const &x : layer->blobs)
    {
      if (x.first == "biases")
      {
	if(x.second->size()>0)
	{
		layerinfo.layerBias = new float[x.second->size()];
		layerinfo.num_biases = x.second->size();
        	for (int biasCount = 0; biasCount < x.second->size(); biasCount++)
        	{
          	layerinfo.layerBias[biasCount] = x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[biasCount]; // put the layer bias in the list
        	}
	}
      }
      else if (x.first == "weights")
      {
	if(x.second->size()>0)
	{
		layerinfo.layerWeights = new float[x.second->size()];
		layerinfo.num_weights = x.second->size();
        	for (int m = 0; m < x.second->size(); m++)
        	{
         	 layerinfo.layerWeights[m] = x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[m];
        	}
	}
      }
    }
    cnnLayersList.push_back(layerinfo); // add the layer information to the List of structure.
    it++;
  }

  print_layerDetails(cnnLayersList);

//  cl::CommandQueue myqueue(mycontext, DeviceList[0]); 	//command queue
//  assert(err==CL_SUCCESS);

//creating  kernel
/*
cl::Kernel Convkernel(program,conv_kernel_name);
assert(err==CL_SUCCESS);
cl::Kernel Maxkernel(program,max_kernel_name);
assert(err==CL_SUCCESS);
cl::Kernel FCLkernel(program,fcl_kernel_name);
assert(err==CL_SUCCESS);
*/
/*
std::ifstream aocx_stream("kernels/"+overlay_name, std::ios::in|std::ios::binary);
checkErr(aocx_stream.is_open() ? CL_SUCCESS:-1, "SimpleKernel.aocx");
std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()+1));

cl::Program program(mycontext, DeviceList, mybinaries);
err=program.build(DeviceList);
assert(err==CL_SUCCESS);
*/

  //cl::CommandQueue *queues[50];
  cl::Buffer *buffers[100];
  for (layersDetails l : cnnLayersList)
  {
    //CNNLayer::Ptr layer = *it;
    //queues[i] = new cl::CommandQueue(mycontext, DeviceList[0]);
	/*
    switch ( l.layerType )
      {
         case "Convolution":
		
            break;
         case "Pooling":
		
            break;
         case "FullyConnected":
		
            break;
         default:
            break;
      }
*/
  } 
return 0;
}
