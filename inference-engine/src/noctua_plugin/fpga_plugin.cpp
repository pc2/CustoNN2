#include "CL/cl.hpp"
#include <inference_engine.hpp>
#include<string.h>
#include<vector>
using namespace InferenceEngine;



unsigned char *images;

void parse_images(std::vector<std::string> imageNames,unsigned char *images,InferenceEngine::CNNNetwork network)
{

InputsDataMap inputInfo = network.getInputsInfo();
if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the plugin **/
inputInfoItem.second->setPrecision(Precision::U8);
inputInfoItem.second->setLayout(Layout::NCHW);

std::vector<std::shared_ptr<unsigned char>> imagesData;
for (auto & i : imageNames) {
   FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
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


tring bitstreamfinder(char* filepath){  
  char * full_filename;
  char * filenameFromPath;
  strtok (filepath,"/");
  while ( (filenameFromPath = strtok (NULL, "/") ) != NULL)
  {
    full_filename = filenameFromPath;
  }
  string str=full_filename;
  size_t lastindex = str.find_last_of("."); 
  string filename = str.substr(0, lastindex);
  filename+=".aocx";
  string str1 = "kernels/"+filename;
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

string overlay_name = bitstreamFinder(model_path);		//Checking the availability of bitstream

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
int no_of_layers = std::distance(it.begin(),it.end());	
	
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
