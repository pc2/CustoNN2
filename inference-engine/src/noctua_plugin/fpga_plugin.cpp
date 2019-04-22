
#include <string>
#include <vector>
#include <iostream>

#include "CL/cl.hpp"
#include "fpga_plugin.h"
#include <format_reader_ptr.h>

#include <inference_engine.hpp>

using namespace InferenceEngine;


//Data structure to store information of each layer
struct layersDetails{
  std::string layerName;
  std::vector<float> layerBias;
  std::vector<float> layerWeights;
};

unsigned char *images;


void parse_images(std::vector<std::string> imageNames,unsigned char *images,InferenceEngine::CNNNetwork network)
{

InputsDataMap inputInfo = network.getInputsInfo();
if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
auto inputInfoItem = *inputInfo.begin();

        // Specifying the precision and layout of input data provided by the user.
        // This should be called before load of the network to the plugin 
inputInfoItem.second->setPrecision(Precision::U8);
inputInfoItem.second->setLayout(Layout::NCHW);

std::vector<std::shared_ptr<unsigned char>> imagesData;
for (auto & i : imageNames) {
   FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                std::cout << "Image " + i + " cannot be read!" << std::endl;
                continue;
            }
            // Store image data 
            std::shared_ptr<unsigned char> data(
                    reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3],
                                    inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

	images = new unsigned char[inputInfoItem.second->getTensorDesc().getDims()[3]*inputInfoItem.second->getTensorDesc().getDims()[2]*imagesData.size()];
	int img_index = 0;
	for(int i=0;i<imagesData.size();i++)
	{
		for(int j=0;j<inputInfoItem.second->getTensorDesc().getDims()[3]*inputInfoItem.second->getTensorDesc().getDims()[2];j++)
		{
			images[img_index] = imagesData.at(i)[j];
			img_index++;
		}
	}

	

}


std::string bitstreamFinder(char* filepath){  

  char * full_filename;
  char * filenameFromPath;
  
  strtok (filepath,"/");
  while ( (filenameFromPath = strtok (NULL, "/") ) != NULL)
  {
    full_filename = filenameFromPath;
  }
  std::string str="";
  str=full_filename;
  size_t lastindex = str.find_last_of("."); 
  std::string filename = str.substr(0, lastindex);
  filename+=".aocx";
  std::string str1 = "kernels/"+filename;
  char char1[20];
  strcpy(char1, str1.c_str());
  FILE *fp = fopen(char1,"r");
  if(fp!=NULL){
    return filename;
  }
  else{
    return "not found";
  }   
}

void fpga_launcher(InferenceEngine::CNNNetwork network, char *model_path,std::vector<std::string> imageNames)
{

std::string overlay_name = bitstreamFinder(model_path);		//Checking the availability of bitstream

parse_images(std::vector<std::string> imageNames,images,network);	

cl_int err;
	
std::vector<cl::Platform> PlatformList;			//Platforms
							
err = cl::Platform::get(&PlatformList);
assert(err==CL_SUCCESS);

std::vector<cl::Device> DeviceList;			//Devices
	
err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList);
assert(err==CL_SUCCESS);

cl::Context mycontext(DeviceList);			//Context
assert(err==CL_SUCCESS);


details::CNNNetworkIterator it(network.actual);
int no_of_layers = static_cast<int>(network.layerCount());

//Get the Layers in the network and store the weights and bias in a structure for each layer.
  
  std::vector<layersDetails> cnnLayersList;
	while (it != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *it;
        layersDetails layerinfo;
        layerinfo.layerName=layer->name;
        int biasCount=0;

        //store the bias and weights 
        for (auto const& x : layer->blobs)
        {
          if(x.first=="biases"){
            layerinfo.layerBias.push_back(x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[biasCount]);   // put the layer bias in the list
            biasCount++;
          }else if(x.first=="weights"){
            for(int m=0;m<x.second->size();m++){
              layerinfo.layerWeights.push_back(x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[m]);
            }             
          }      
        }
    cnnLayersList.push_back(layerinfo);  // add the layer information to the List of structure.
  }



	
//cl::CommandQueue myqueue(mycontext, DeviceList[0]); 	//command queue
//assert(err==CL_SUCCESS);

cl::CommandQueue *queues[50];
cl::Buffer *buffers[100];
for(int i=0;i<no_of_layers;i++)
{
	CNNLayer::Ptr layer = *it;
	queues[i] = new cl::CommandQueue(mycontext,DeviceList[0]);
	
	
}

}
