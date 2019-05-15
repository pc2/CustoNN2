#include <string>
#include <vector>
#include <array>
#include <iostream>
#include<fstream>
#include <math.h>
#include "CL/cl.hpp"
#include "fpga_plugin.h"
#include <format_reader_ptr.h>
#include <chrono>
#include <thread>
#include <assert.h> 

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
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
int num_images,dim_x,dim_y;
void printImage(unsigned char * image,int numberOfImages,int xdim,int ydim){

	std::cout<<"Image Pixels are:"<<std::endl;
	for(int m=0;m<numberOfImages;m++){
		for(int i=0;i<xdim;i++){
			for(int k=0;k<ydim;k++){
				std::cout<<unsigned(image[(m*xdim*ydim)+(i*ydim)+k]) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}

}

void printFilterWeights(float * layerWeights){
	std::cout<<"1st Filter weights are:"<<std::endl;
	for(int i=0;i<5;i++){
		for(int k=0;k<5;k++){
			std::cout<<layerWeights[i*5+k] << " ";
		}
		std::cout<<std::endl;
	}


}


void parse_images(std::vector<std::string> imageNames, InferenceEngine::CNNNetwork network)
{

  InputsDataMap inputInfo = network.getInputsInfo();
  
	if (inputInfo.size() != 1)
    throw std::logic_error("Sample supports topologies only with 1 input");
	
  auto inputInfoItem = *inputInfo.begin();
	inputInfoItem.second->setPrecision(Precision::U8);
  inputInfoItem.second->setLayout(Layout::NCHW);
	std::vector<std::shared_ptr<unsigned char>> imagesData;
	std::string input_name = network.getInputsInfo().begin()->first;
	int imageIndex=0;
	
  images = new unsigned char[inputInfoItem.second->getTensorDesc().getDims()[3]*inputInfoItem.second->getTensorDesc().getDims()[2]*imageNames.size()*inputInfoItem.second->getTensorDesc().getDims()[1]];

for(std::string i: imageNames){
	std::cout<<"Reading Image :"<<i<<std::endl;
	cv::Mat image = cv::imread(i);

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
  num_images = imagesData.size();
	dim_x = inputInfoItem.second->getTensorDesc().getDims()[3];
	dim_y = inputInfoItem.second->getTensorDesc().getDims()[2];
	size_t channels_number = inputInfoItem.second->getTensorDesc().getDims()[1];
  size_t image_size = inputInfoItem.second->getTensorDesc().getDims()[3] * inputInfoItem.second->getTensorDesc().getDims()[2];
	

  if (imagesData.empty())
    throw std::logic_error("Valid input images were not found!");
	std::cout<<"Number of Pixels:"<<image_size*channels_number<<" , Number of Images:"<<imagesData.size() <<std::endl;
  cv::resize(image, image, cv::Size(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]));

	
		for (size_t pid = 0; pid < image_size; ++pid) {
			for (size_t ch = 0; ch < channels_number; ++ch) {
					images[(imageIndex*image_size*channels_number) + ch * image_size + pid] = image.at<cv::Vec3b>(pid)[ch];
			}
		}
	imageIndex++;
	}
	std::cout<<"Images Pixel: " <<std::endl;
	printImage(images,num_images,dim_x,dim_y);
  
	
  
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
  std::cout <<"Finding the bitstream"<< std::endl;
  char *full_filename;
  char *filenameFromPath;

  strtok(filepath, "/");
  while ((filenameFromPath = strtok(NULL, "/")) != NULL)
  {
    full_filename = filenameFromPath;
  }
  std::string str = "";
  str = full_filename;
  std::cout<<" AOCX File is:" <<str<<std::endl;
  size_t lastindex = str.length();
  std::string filename = str.substr(0, lastindex);
  filename += ".aocx";
  std::string str1 = "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs/dldt/kernels/" + filename;
  std::cout<<"Final AOCX File is:" <<str1<<std::endl;
  char *char1= new char[str1.length()+1];
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
  //std::string overlay_name = bitstreamFinder(model_path); //Checking the availability of bitstream
	std::string overlay_name = " ";
  if(overlay_name=="not found"){
    std::cout<<" Bitstream not found\n";
   //return -1; 
    //exit(0);  
  }else{
    std::cout<<" Bitstream found\n";
  } 
 // parse_images(imageNames, network);

  cl_int err;

  std::vector<cl::Platform> PlatformList; //Platforms

  err = cl::Platform::get(&PlatformList);
  assert(err == CL_SUCCESS);
	std::cout<<" Error code after Get Platform:"<<" is ===>"<<err<<std::endl;
   for (int i=0; i<PlatformList.size(); i++)
        {
                printf("Platform Number: %d\n", i);
                std::cout << "Platform Name: "<<PlatformList.at(i).getInfo<CL_PLATFORM_NAME>()<<"\n";
                std::cout << "Platform Profile: "<<PlatformList.at(i).getInfo<CL_PLATFORM_PROFILE>()<<"\n";
                std::cout << "Platform Version: "<<PlatformList.at(i).getInfo<CL_PLATFORM_VERSION>()<<"\n";
                std::cout << "Platform Vendor: "<<PlatformList.at(i).getInfo<CL_PLATFORM_VENDOR>()<<"\n\n";
        }



  std::vector<cl::Device> DeviceList_Master,DeviceList; //Devices

  err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList_Master);
  assert(err == CL_SUCCESS);
	std::cout<<" Error code after Get Device:"<<" is ===>"<<err<<std::endl;
	DeviceList.push_back(DeviceList_Master[0]);
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
	std::cout<<" Error code after Context:"<<" is ===>"<<err<<std::endl;
  
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
		int inLayer=0;
		for (auto& src : layer->insData) {
			if(layer->name!="input"){
				auto previousLayer = layer->insData[inLayer].lock()->getCreatorLayer().lock();
				std::string inputLayerName = previousLayer->name;
				std::cout<<"		Input:" <<inputLayerName<<std::endl;
				inLayer++;
			}
			
		}
		int outLayer=0;
		for (auto it : layer->outData[0]->getInputTo()) {
                std::cout<<"		Output:" <<it.second->name<<std::endl;
								outLayer++;
            }

		std::cout<<std::endl;
		std::cout<<std::endl;
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

 cl::CommandQueue myqueue(mycontext, DeviceList[0]); 	//command queue
  assert(err==CL_SUCCESS);
	std::cout<<" Error code after Cmd Queue:"<<" is ===>"<<err<<std::endl;

std::ifstream aocx_stream("/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/simplecnn_openvino/lenet_iter_10000.aocx", std::ios::in|std::ios::binary);
//std::ifstream aocx_stream("/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs/dldt/kernels/lenet_iter_10000.aocx", std::ios::in|std::ios::binary);
//checkErr(aocx_stream.is_open() ? CL_SUCCESS:-1, overlay_name);
std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()+1));


cl::Program program(mycontext, DeviceList, mybinaries);                                     

err=program.build(DeviceList);
assert(err==CL_SUCCESS);
std::cout<<" Error code after BUILD:"<<" is ===>"<<err<<std::endl;
cl::Kernel *kernels[52];
int kernel_index = 0;
  //cl::CommandQueue *queues[50];
  cl::Buffer *buffers[100];
  int buffer_index = 0;
  //std::cout<<"Images array size: "<<sizeof(images)/sizeof(images[0])<<"\n";
	std::cout<<"first image\n";
	//for(int i=0;i<784;i++)
		//std::cout<<unsigned(images[i])<<"\n";
	
  buffers[buffer_index] = new cl::Buffer(mycontext, CL_MEM_READ_ONLY, sizeof(cl_uchar)*dim_x*dim_y*num_images);
  err = myqueue.enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_uchar)*dim_x*dim_y*num_images, images); //images buffer
  assert(err==CL_SUCCESS);
 err= myqueue.finish(); 
 std::cout<<" Error code after image transfer:"<<kernel_index<<" is ===>"<<err<<std::endl;
 assert(err==CL_SUCCESS);
  std::cout<<"images copied\n";
  int num_filters = 0;
  int num_classes = 0;
  int num_pixels = dim_x * dim_y;
  for (layersDetails l : cnnLayersList)
  {
    if(l.layerType == "Convolution")
	{
		kernels[kernel_index] = new cl::Kernel(program, "ConvolutionLayer", &err);
		assert(err==CL_SUCCESS);
		
		err = kernels[kernel_index]->setArg(0,*buffers[buffer_index]);   //first argument, input, also the output of the previous layer
		assert(err==CL_SUCCESS);
		buffer_index++;
		std::cout<<"images passed\n";		
		
		printFilterWeights(l.layerWeights);
		buffers[buffer_index] = new cl::Buffer(mycontext,CL_MEM_READ_ONLY, sizeof(cl_float)*l.num_weights);
		err = myqueue.enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float)*l.num_weights, l.layerWeights);    //weights
		err = myqueue.finish();
		std::cout<<" Error code after weights transfer:"<<kernel_index<<" is ===>"<<err<<std::endl;
		assert(err==CL_SUCCESS);		
		err = kernels[kernel_index]->setArg(1,*buffers[buffer_index]);
		assert(err==CL_SUCCESS);
		buffer_index++;
		
		std::cout<<"weights passed\n";

		buffers[buffer_index] = new cl::Buffer(mycontext,CL_MEM_READ_ONLY, sizeof(cl_float)*l.num_biases);
		err = myqueue.enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float)*l.num_biases, l.layerBias);           //biases
		err = myqueue.finish();
		std::cout<<" Error code after bias transfer:"<<kernel_index<<" is ===>"<<err<<std::endl;
		assert(err==CL_SUCCESS);
		err = kernels[kernel_index]->setArg(2,*buffers[buffer_index]);
		assert(err==CL_SUCCESS);
		buffer_index++;
		std::cout<<"biases passed\n";

		int f_dim = l.params["kernel"].at(0) - '0';
		std::cout<<"num of filter rows : "<<f_dim<<"\n";
		err = kernels[kernel_index]->setArg(3,f_dim);		//filter rows
		assert(err==CL_SUCCESS);
		std::cout<<"filter rows passed\n";		

		err = kernels[kernel_index]->setArg(4,(int)f_dim);		//filter cols
		assert(err==CL_SUCCESS);
		std::cout<<"filter cols passed\n";

		 num_filters = std::atoi(l.params["output"].c_str());
		std::cout<<"no of filters from the map: "<<num_filters<<"\n";
		err = kernels[kernel_index]->setArg(5,num_filters);		//no of filters
		assert(err==CL_SUCCESS);
		std::cout<<"no of filters passed\n";

		err = kernels[kernel_index]->setArg(6,num_images);		//no of images
		assert(err==CL_SUCCESS);
		std::cout<<"no of images passed\n";

		err = kernels[kernel_index]->setArg(7,dim_x);		//no of image rows
		assert(err==CL_SUCCESS);
		std::cout<<"no of image rows passed\n";	
	
		err = kernels[kernel_index]->setArg(8,dim_y);		//no of image cols
		assert(err==CL_SUCCESS);
		std::cout<<"no of image cols passed\n";
		
		int pad_begin[] = {l.params["pads_begin"].at(0) - '0',l.params["pads_begin"].at(2) - '0'};
		buffers[buffer_index] = new cl::Buffer(mycontext,CL_MEM_READ_ONLY, sizeof(cl_int)*2);
		err = myqueue.enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_int)*2,pad_begin);           //pad begin
		assert(err==CL_SUCCESS);
		err = myqueue.finish();
		assert(err==CL_SUCCESS);
		//std::cout<<"padding from the map"<<pad<<"\n";
		err = kernels[kernel_index]->setArg(9,*buffers[buffer_index]);		//padding
		assert(err==CL_SUCCESS);
		std::cout<<"padding passed\n";
		buffer_index++;
		
		int pad_end[] = {l.params["pads_end"].at(0) - '0',l.params["pads_end"].at(2) - '0'};
		buffers[buffer_index] = new cl::Buffer(mycontext,CL_MEM_READ_ONLY, sizeof(cl_int)*2);
		err = myqueue.enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_int)*2,pad_end);           //pad end
		myqueue.finish();
		//std::cout<<"padding from the map"<<pad<<"\n";
		err = kernels[kernel_index]->setArg(10,*buffers[buffer_index]);		//padding
		assert(err==CL_SUCCESS);
		std::cout<<"padding passed\n";
		buffer_index++;

		int stride = l.params["strides"].at(0) - '0';
		std::cout<<"stride from the map(conv): "<<stride<<"\n";
		err = kernels[kernel_index]->setArg(11,stride);		//stride
		assert(err==CL_SUCCESS);
		std::cout<<"stride passed\n";
		//err = kernels[kernel_index]->setArg(9,dim_x);		//conv output rows
		//assert(err==CL_SUCCESS);

		//err = kernels[kernel_index]->setArg(10,dim_y);		//conv output cols
		//assert(err==CL_SUCCESS);
		
		buffers[buffer_index] = new cl::Buffer(mycontext,CL_MEM_READ_WRITE,sizeof(cl_double)*dim_x*dim_y*num_images*num_filters);
		err = kernels[kernel_index]->setArg(12,*buffers[buffer_index]);								//output of conv
		assert(err==CL_SUCCESS);
		std::cout<<"conv output passed\n";

		err=myqueue.enqueueTask(*kernels[kernel_index]);
		//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
		assert(err==CL_SUCCESS);
		std::cout<<" Error code soon after conv layer for kernel:"<<kernel_index<<" is ===>"<<err<<std::endl;
		kernel_index++;
		
		err=myqueue.finish();
		assert(err==CL_SUCCESS);

		std::cout<<" Error code  after conv layer finish for kernel:"<<kernel_index<<" is ===>"<<err<<std::endl;

		num_pixels *= num_filters;
	}
	else if(l.layerType == "Pooling")
	{
		kernels[kernel_index] = new cl::Kernel(program, "MaxPool", &err);
		assert(err==CL_SUCCESS);
		
		err = kernels[kernel_index]->setArg(0,*buffers[buffer_index]);   //first argument, input, also the output of the previous layer
		assert(err==CL_SUCCESS);
		buffer_index++;
		std::cout<<"images passed\n";
		err = kernels[kernel_index]->setArg(1,dim_x);		//conv output rows
		assert(err==CL_SUCCESS);
		std::cout<<"conv output rows passed\n";
		err = kernels[kernel_index]->setArg(2,dim_y);		//conv output cols
		assert(err==CL_SUCCESS);
		std::cout<<"conv output cols passed\n";
		err = kernels[kernel_index]->setArg(3,num_filters);		//no of filters
		assert(err==CL_SUCCESS);
		std::cout<<"no of filters passed\n";
		int stride = l.params["strides"].at(0) - '0';
		std::cout<<"Stride from the map: "<<stride<<"\n";
		err = kernels[kernel_index]->setArg(4,stride);		//stride
		assert(err==CL_SUCCESS);
		std::cout<<"stride passed: "<<stride<<"\n";
		err = kernels[kernel_index]->setArg(5,num_images);		//no of images
		assert(err==CL_SUCCESS);
		std::cout<<"no of images passed\n";
		//err = kernels[kernel_index]->setArg(6,dim_y);		//no of image cols
		//assert(err==CL_SUCCESS);
		dim_x /= stride;
		dim_y /= stride;
		num_pixels /= (stride * stride);
		std::cout<<"dim x: "<<dim_x<<" dim y: "<<dim_y<<" num_pixels: "<<num_pixels<<"\n";		
	
		buffers[buffer_index] = new cl::Buffer(mycontext,CL_MEM_READ_WRITE,sizeof(cl_double)*dim_x*dim_y*num_images*num_filters);
		err = kernels[kernel_index]->setArg(6,*buffers[buffer_index]);								//output of pool
		assert(err==CL_SUCCESS);
		std::cout<<"pool output\n";
		err=myqueue.enqueueTask(*kernels[kernel_index]);
		//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
		assert(err==CL_SUCCESS);
		kernel_index++;
		
		//dim_x /= stride;
		//dim_y /= stride;
		//num_pixels /= (stride * stride);
	}
	else if(l.layerType == "FullyConnected")
	{
		kernels[kernel_index] = new cl::Kernel(program, "FCL_Kernel", &err);
		assert(err==CL_SUCCESS);

		err = kernels[kernel_index]->setArg(0,*buffers[buffer_index]);   //first argument, input, also the output of the previous layer
		assert(err==CL_SUCCESS);
		buffer_index++;
		std::cout<<"images passed\n";		
		
		buffers[buffer_index] = new cl::Buffer(mycontext,CL_MEM_READ_ONLY, sizeof(cl_float)*l.num_weights);
		err = myqueue.enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float)*l.num_weights, l.layerWeights);    //weights
		myqueue.finish();
		assert(err==CL_SUCCESS);		
		err = kernels[kernel_index]->setArg(1,*buffers[buffer_index]);
		assert(err==CL_SUCCESS);
		buffer_index++;
		std::cout<<"weights passed\n";
		buffers[buffer_index] = new cl::Buffer(mycontext,CL_MEM_READ_ONLY, sizeof(cl_float)*l.num_biases);
		err = myqueue.enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float)*l.num_biases, l.layerBias);           //biases
		myqueue.finish();
		assert(err==CL_SUCCESS);
		err = kernels[kernel_index]->setArg(2,*buffers[buffer_index]);
		assert(err==CL_SUCCESS);
		buffer_index++;
		std::cout<<"biases passed\n";
		err = kernels[kernel_index]->setArg(3,num_pixels);		//no of pixels
		assert(err==CL_SUCCESS);
		std::cout<<"no of pixels passed\n";
		 num_classes = std::atoi(l.params["out-size"].c_str());
		std::cout<<"num classes from the map: "<<num_classes<<"\n";
		err = kernels[kernel_index]->setArg(4,num_classes);		//no of classes
		assert(err==CL_SUCCESS);	
		std::cout<<"no of classes passed\n";
		err = kernels[kernel_index]->setArg(5,num_images);		//no of images
		assert(err==CL_SUCCESS);
		std::cout<<"no of images passed\n";
		buffers[buffer_index] = new cl::Buffer(mycontext,CL_MEM_READ_WRITE,sizeof(cl_int)*num_images);
		err = kernels[kernel_index]->setArg(6,*buffers[buffer_index]);								//output of FC
		assert(err==CL_SUCCESS);
		std::cout<<"output of FC passed\n";
		err = kernels[kernel_index]->setArg(7,dim_x);		//rows
		assert(err==CL_SUCCESS);
		std::cout<<"no of rows passed\n";
		err = kernels[kernel_index]->setArg(8,dim_y);		//cols
		assert(err==CL_SUCCESS);
		std::cout<<"no of cols passed\n";
		err = kernels[kernel_index]->setArg(9,num_filters);		//no of filters
		assert(err==CL_SUCCESS);
		std::cout<<"no of filters passed\n";
		err=myqueue.enqueueTask(*kernels[kernel_index]);
		assert(err==CL_SUCCESS);
		kernel_index++;
	}
  } 

int final_labels[num_images];
err=myqueue.enqueueReadBuffer(*buffers[buffer_index], CL_TRUE, 0, sizeof(cl_int)*num_images, final_labels);
assert(err==CL_SUCCESS);

err=myqueue.finish();
assert(err==CL_SUCCESS);

for(int i=0;i<num_images;i++)
{
	std::cout<<"Image "<<imageNames[i]<<" : "<<final_labels[i]<<"\n";

}


return 0;
}