#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <math.h>
#include "CL/cl.hpp"
#include "fpga_plugin.h"
#include <format_reader_ptr.h>
#include <chrono>
#include <thread>
#include <assert.h>
#include <ie_builders.hpp>
#include <queue>
#include <sys/types.h>
#include <dirent.h>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>


// Directories where the aocx files will be stored
#define GoogLeNet_DIR "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/GoogLeNet"
#define ResNet_DIR "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/ResNet"

using namespace InferenceEngine;
using namespace boost;
using namespace boost::property_tree;

std::string supported_layers[4] = {"Convolution", "Pooling", "FullyConnected", "Concat"};
std::map<std::string, int> layerIDMap;
std::vector<int> ID_list;

/**
 * Data structure to store information of each layer
 */
struct layersDetails
{
	int layerID;
	std::string layerName;
	std::string layerType;
	float *layerBias;
	float *layerWeights;
	int num_biases;
	int num_weights;
	std::map<std::string, std::string> params;
	std::vector<std::string> inputLayerNames;
	std::vector<std::string> outputLayerNames;
	std::vector<struct layersDetails *> children;
	int device_num;
};

unsigned char *images;
int num_images, dim_x, dim_y;


bool isLayerSupported(std::string layer_name)
{
	for (int i = 0; i < 4; i++)
	{
		if (layer_name.compare(supported_layers[i]) == 0)
			return true;
	}

	return false;
}

bool isDuplicate(int id)
{
	for (std::vector<int>::iterator it = ID_list.begin(); it != ID_list.end(); ++it)
	{
		if (id == *it)
		{
			return true;
		}
	}

	return false;
}

/**
 * Function to print Images
 */
void printImage(unsigned char *image, int numberOfImages, int xdim, int ydim)
{

	std::cout << "Image Pixels are:" << std::endl;
	for (int m = 0; m < numberOfImages; m++)
	{
		for (int i = 0; i < xdim; i++)
		{
			for (int k = 0; k < ydim; k++)
			{
				std::cout << unsigned(image[(m * xdim * ydim) + (i * ydim) + k]) << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

/**
 * Printing the Filter weights
 */
void printFilterWeights(float *layerWeights)
{
	std::cout << "1st Filter weights are:" << std::endl;
	for (int i = 0; i < 5; i++)
	{
		for (int k = 0; k < 5; k++)
		{
			std::cout << layerWeights[i * 5 + k] << " ";
		}
		std::cout << std::endl;
	}
}

/**
 * Parse the input images
 */
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
	int imageIndex = 0;

	images = new unsigned char[inputInfoItem.second->getTensorDesc().getDims()[3] * inputInfoItem.second->getTensorDesc().getDims()[2] * imageNames.size() * inputInfoItem.second->getTensorDesc().getDims()[1]];

	for (std::string i : imageNames)
	{
		std::cout << "Reading Image :" << i << std::endl;
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
		std::cout << "Number of Pixels:" << image_size * channels_number << " , Number of Images:" << imagesData.size() << std::endl;
		cv::resize(image, image, cv::Size(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]));

		for (size_t pid = 0; pid < image_size; ++pid)
		{
			for (size_t ch = 0; ch < channels_number; ++ch)
			{
				images[(imageIndex * image_size * channels_number) + ch * image_size + pid] = image.at<cv::Vec3b>(pid)[ch];
			}
		}
		imageIndex++;
	}
	std::cout << "Images Pixel: " << std::endl;
	printImage(images, num_images, dim_x, dim_y);
}

/**
 * Print the Layer Details from the layersDetails structure.
 */
void print_layersDetails(std::vector<layersDetails> cnnlayers)
{

	for (layersDetails a : cnnlayers)
	{
		std::cout << "LayerID:"    << a.layerID << std::endl;
		std::cout << "LayerName: " << a.layerName << std::endl;
		std::cout << "LayerType: " << a.layerType << std::endl;
		std::cout << "Bias:";
		for (int i = 0; i < a.num_biases; i++)
		{
			std::cout << a.layerBias[i] << " ";
		}
		std::cout << std::endl;

		/*
		std::cout << "Weights:";
		for (int i = 0; i < a.num_weights; i++)
		{
			//std::cout << a.layerWeights[i] << " ";
		}
		*/
		std::cout << "\t Layer Parameters: " << std::endl;
		for (auto const &y : a.params)
		{
			std::cout << "\t" << y.first << ":" << y.second << std::endl;
		}

		std::cout << "Input Layers: " << std::endl;
		for (auto const &inputLayer : a.inputLayerNames)
		{
			std::cout << "\t" << inputLayer << std::endl;
		}

		std::cout << "Output Layers: " << std::endl;
		for (auto const &outputLayer : a.outputLayerNames)
		{
			std::cout << "\t" << outputLayer << std::endl;
		}
	}
	std::cout << std::endl;
}

/**
 * function to check if the input model's bitstream is present in the Noctua FPGA Bitstream Repo.
 */
std::string bitstreamFinder(char *filepath)
{
	std::cout << "Finding the bitstream" << std::endl;
	char *full_filename;
	char *filenameFromPath;

	strtok(filepath, "/");
	while ((filenameFromPath = strtok(NULL, "/")) != NULL)
	{
		full_filename = filenameFromPath;
	}
	std::string str = "";
	str = full_filename;
	std::cout << " AOCX File is:" << str << std::endl;
	size_t lastindex = str.length();
	std::string filename = str.substr(0, lastindex);
	filename += ".aocx";
	std::string str1 = "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs/dldt/kernels/" + filename;
	std::cout << "Final AOCX File is:" << str1 << std::endl;
	char *char1 = new char[str1.length() + 1];
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

struct layersDetails *parse_root(InferenceEngine::CNNNetwork network)
{
	struct layersDetails *root = new layersDetails;
	details::CNNNetworkIterator it(network.actual);
	int no_of_layers = static_cast<int>(network.layerCount());

	//Parse CNNNetwork to Network to get the IDs
	std::cout << "inside parse root\n";

	//Get the Layers in the network and store the weights and bias in a structure for each layer.
	//std::vector<layersDetails> cnnLayersList;
	while (it != details::CNNNetworkIterator())
	{
		//std::cout<<
		CNNLayer::Ptr layer = *it;
		if (isLayerSupported(layer->type))
		{
			//layersDetails layerinfo;
			root->layerName = layer->name;
			root->layerType = layer->type;
			root->params = layer->params;
			root->num_biases = 0;
			root->num_weights = 0;
			std::cout << "Root name: " << root->layerName << "\n";
			//Search the Layer ID in the Map
			auto search = layerIDMap.find(layer->name);
			if (search != layerIDMap.end())
			{
				root->layerID = search->second;
			}
			else
			{
				std::cerr << " Layer ID of the layer: '" << layer->name << "' NOT FOUND" << std::endl;
				//return -1;
			}
			std::cout << "Parsing Kernel:"
					  << "  --- " << layer->name << std::endl;
			//Insert the Input Layer Names into the vector
			int inLayer = 0;
			for (auto &src : layer->insData)
			{
				if (layer->name != "input")
				{
					auto previousLayer = layer->insData[inLayer].lock()->getCreatorLayer().lock();
					std::string inputLayerName = previousLayer->name;
					root->inputLayerNames.push_back(inputLayerName);
					inLayer++;
				}
			}

			//Insert the Output Layer Names into the vector
			int outLayer = 0;
			for (auto it : layer->outData[0]->getInputTo())
			{
				root->outputLayerNames.push_back(it.second->name);
				outLayer++;
			}

			//store the bias and weights
			for (auto const &x : layer->blobs)
			{
				if (x.first == "biases")
				{
					if (x.second->size() > 0)
					{
						root->layerBias = new float[x.second->size()];
						root->num_biases = x.second->size();
						for (int biasCount = 0; biasCount < x.second->size(); biasCount++)
						{
							root->layerBias[biasCount] = x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[biasCount]; // put the layer bias in the list
						}
					}
				}
				else if (x.first == "weights")
				{
					if (x.second->size() > 0)
					{
						root->layerWeights = new float[x.second->size()];
						root->num_weights = x.second->size();
						for (int m = 0; m < x.second->size(); m++)
						{
							root->layerWeights[m] = x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[m];
						}
					}
				}
			}
			return root;
			//break;
		}
		it++;
	}

	return NULL;
}

// Level order traversal to find node with particular ID
struct layersDetails *findbyID(struct layersDetails *root, int id)
{
	if (root == NULL)
		return NULL;

	std::queue<struct layersDetails *> q;

	q.push(root);

	while (!q.empty())
	{
		int n = q.size();

		while (n > 0)
		{

			struct layersDetails *p = q.front();
			q.pop();
			if (p != NULL)
			{
				if (p->layerID == id)
					return p;
			}
			// Enqueue all children of the dequeued item
			for (std::vector<struct layersDetails *>::iterator it = p->children.begin(); it != p->children.end(); ++it)
			{
				if (*it != NULL)
					q.push(*it);
			}
			n--;
		}
	}

	return NULL;
}

struct layersDetails *parse_child(InferenceEngine::CNNNetwork network, std::string layer_name, struct layersDetails *root)
{
	const char *l_name = layer_name.c_str();
	std::cout << "For layer name: " << l_name << "\n";
	CNNLayerPtr layer = network.getLayerByName(l_name);
	std::cout << "Inside parse child\n";
	if (isLayerSupported(layer->type))
	{
		auto search = layerIDMap.find(layer_name);
		int ID = search->second;
		if (isDuplicate(ID))
		{
			struct layersDetails *child = findbyID(root, ID);
			return child;
		}
		else
		{

			struct layersDetails *child = new layersDetails;
			//retrieving the child information
			child->layerName = layer->name;
			child->layerType = layer->type;
			child->params = layer->params;
			child->num_biases = 0;
			child->num_weights = 0;

			//Search the Layer ID in the Map
			auto search = layerIDMap.find(layer->name);
			if (search != layerIDMap.end())
			{
				child->layerID = search->second;
			}
			else
			{
				std::cerr << " Layer ID of the layer: '" << layer->name << "' NOT FOUND" << std::endl;
				//return -1;
			}

			ID_list.push_back(child->layerID);
			//Insert the Input Layer Names into the vector
			int inLayer = 0;
			for (auto &src : layer->insData)
			{
				if (layer->name != "input")
				{
					auto previousLayer = layer->insData[inLayer].lock()->getCreatorLayer().lock();
					std::string inputLayerName = previousLayer->name;
					child->inputLayerNames.push_back(inputLayerName);
					inLayer++;
				}
			}

			int outLayer = 0;
			for (auto it : layer->outData[0]->getInputTo())
			{
				child->outputLayerNames.push_back(it.second->name);
				outLayer++;
			}

			//store the bias and weights
			for (auto const &x : layer->blobs)
			{
				if (x.first == "biases")
				{
					if (x.second->size() > 0)
					{
						child->layerBias = new float[x.second->size()];
						child->num_biases = x.second->size();
						for (int biasCount = 0; biasCount < x.second->size(); biasCount++)
						{
							child->layerBias[biasCount] = x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[biasCount]; // put the layer bias in the list
						}
					}
				}
				else if (x.first == "weights")
				{
					if (x.second->size() > 0)
					{
						child->layerWeights = new float[x.second->size()];
						child->num_weights = x.second->size();
						for (int m = 0; m < x.second->size(); m++)
						{
							child->layerWeights[m] = x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[m];
						}
					}
				}
			}

			for (std::vector<std::string>::iterator it = child->outputLayerNames.begin(); it != child->outputLayerNames.end(); ++it)
			{
				child->children.push_back(parse_child(network, *it, root));
			}

			return child;
		}
	}
	else
	{
		int outLayer = 0;
		struct layersDetails temp;
		for (auto it : layer->outData[0]->getInputTo())
		{
			temp.outputLayerNames.push_back(it.second->name);
			outLayer++;
		}
		//std::cout<<"Unsupported Layer name: "<<layer->type<<" outputs vector size: "<<temp.outputLayerNames.size()<<" Child of this layer "<<temp.outputLayerNames.front()<<"\n";
		if (temp.outputLayerNames.size() > 0)
			return parse_child(network, temp.outputLayerNames.front(), root);
		else
			return NULL;
	}
}

void printCNNTree(layersDetails *root)
{
	std::cout << " Printing CNN Tree..." << std::endl;
	std::queue<struct layersDetails *> q;

	q.push(root);
	while (!q.empty())
	{
		int n = q.size();

		while (n > 0)
		{
			struct layersDetails *p = q.front();
			q.pop();
			if (p != NULL)
			{
				std::cout << p->layerID << " -- " << p->layerName << " -- " << p->layerType << " Device num: " << p->device_num << std::endl;
			}
			// Enqueue all children of the dequeued item
			for (std::vector<struct layersDetails *>::iterator it = p->children.begin(); it != p->children.end(); ++it)
			{
				if (*it != NULL)
					q.push(*it);
			}
			n--;
		}
	}
}

void printPlatforms(std::vector<cl::Platform> PlatformList)
{
	//Printing the Plaforms
	for (int i = 0; i < PlatformList.size(); i++)
	{
		printf("Platform Number: %d\n", i);
		std::cout << "Platform Name: " << PlatformList.at(i).getInfo<CL_PLATFORM_NAME>() << "\n";
		std::cout << "Platform Profile: " << PlatformList.at(i).getInfo<CL_PLATFORM_PROFILE>() << "\n";
		std::cout << "Platform Version: " << PlatformList.at(i).getInfo<CL_PLATFORM_VERSION>() << "\n";
		std::cout << "Platform Vendor: " << PlatformList.at(i).getInfo<CL_PLATFORM_VENDOR>() << "\n\n";
	}
}

void printDevices(std::vector<cl::Device> DeviceList1)
{

	for (int i = 0; i < DeviceList1.size(); i++)
	{
		printf("Device Number: %d\n", i);
		std::cout << "Device Name: " << DeviceList1.at(i).getInfo<CL_DEVICE_NAME>() << "\n";
		std::cout << "Device Vendor: " << DeviceList1.at(i).getInfo<CL_DEVICE_VENDOR>() << "\n";
		std::cout << "Is Device Available?: " << DeviceList1.at(i).getInfo<CL_DEVICE_AVAILABLE>() << "\n";
		std::cout << "Is Device Little Endian?: " << DeviceList1.at(i).getInfo<CL_DEVICE_ENDIAN_LITTLE>() << "\n";
		std::cout << "Device Max Compute Units: " << DeviceList1.at(i).getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
		std::cout << "Device Max Work Item Dimensions: " << DeviceList1.at(i).getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << "\n";
		std::cout << "Device Max Work Group Size: " << DeviceList1.at(i).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
		std::cout << "Device Max Frequency: " << DeviceList1.at(i).getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "\n";
		std::cout << "Device Max Mem Alloc Size: " << DeviceList1.at(i).getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << "\n\n";
	}
}

const ptree& empty_ptree(){
    static ptree t;
    return t;
}

void xml_parser(char *filename,std::vector<std::string> kernel_names)
{
	ptree tree;
	read_xml(filename, tree);
	const ptree & formats = tree.get_child("board", empty_ptree());
	BOOST_FOREACH(const ptree::value_type & f, formats){
        std::string at = f.first + ".<xmlattr>";
        const ptree & attributes = formats.get_child(at, empty_ptree());
        //cout << "Extracting attributes from " << at << ":" << endl;
        BOOST_FOREACH(const ptree::value_type &v, attributes)
	{
		if(v.first == "name")
		{
			//std::cout << "First: " << v.first.data() << " Second: " << v.second.data() << std::endl;
			kernel_names.push_back(v.second.data);
		}
        }
    }
	

}

/**
 * OPENVINO FPGA NOCTUA PLUGIN is implemented in this function  
 */
int fpga_launcher(InferenceEngine::CNNNetwork network, char *model_path, std::vector<std::string> imageNames, std::string model_name,int rank)
{
	std::cout << "In FPGA Launcher" << std::endl;
	//std::string overlay_name = bitstreamFinder(model_path); //Checking the availability of bitstream
	std::string overlay_name = " ";
	if (overlay_name == "not found")
	{
		std::cout << " Bitstream not found\n";
		//return -1;
		//exit(0);
	}
	else
	{
		std::cout << " Bitstream found\n";
	}
	parse_images(imageNames, network);

	cl_int err;

	std::vector<cl::Platform> PlatformList; //Platforms

	err = cl::Platform::get(&PlatformList);
	assert(err == CL_SUCCESS);
	std::cout << " Error code after Get Platform:"
			  << " is ===>" << err << std::endl;

	printPlatforms(PlatformList);

	std::vector<cl::Device> DeviceList_Master,DeviceList1,DeviceList2;
	//std::vector<std::vector<cl::Device>> Devices_to_flash; //Devices
	//Printing the Devices available for the given platform.
	err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList_Master);
	assert(err == CL_SUCCESS);
	std::cout << " Error code after Get Device is ===> " << err << std::endl;

	//Adding the first device to a seperate List
	DeviceList1.push_back(DeviceList_Master[0]);
	DeviceList2.push_back(DeviceList_Master[1]);
	//Printing the Devices

	printDevices(DeviceList_Master);
	
	
	cl::Context context1(DeviceList1);
	cl::Context context2(DeviceList2);
	//cl::Context mycontext(DeviceList1); //Context
	//cl::Context mycontext1(DeviceList2);
	cl::CommandQueue *cmd_queues[250]; // To be dynamically allocated at kernel launch, one per kernel. the index  of cmd queue array is Layer ID.
	

	//assert(err == CL_SUCCESS);
	//std::cout << " Error code after Context:"<< " is ===>" << err << std::endl;

	Builder::Network originalNetwork(network);

	// This map is to store LayerName <-> LayerID pair

	//Insert values the key-value pair
	for (const auto &layer : originalNetwork.getLayers())
	{
		//std:: cout<<layer.getId()<<" "<<layer.getName()<<" "<<layer.getType()<<" "<<layer.getParameters().size()<<std::endl;
		layerIDMap.insert(std::pair<std::string, int>(layer.getName(), layer.getId()));
	}
	std::cout << std::endl;

	struct layersDetails *root = parse_root(network);
	//root node obtained to which images will go as input

	//To obtain the rest of the tree structure
	//Iterating over the outputs of root node, i.e. it's children

	if (root != NULL)
	{
		for (std::vector<std::string>::iterator it = root->outputLayerNames.begin(); it != root->outputLayerNames.end(); ++it)
		{
			root->children.push_back(parse_child(network, *it, root));
		}
	}

	std::cout << "Number of children of root: " << root->children.size() << "\n";
	
	// Code for assigning affinity to layers, device numbering starts at 0
	

	// Code for Affinity ends

	printCNNTree(root);
	
	std::string file1 = GoogLeNet_DIR+"Inception"+std::to_string(2*rank)+".aocx";     //first aocx
	std::string file2 = GoogLeNet_DIR+"Inception"+std::to_string(2*rank+1)+".aocx";   //second aocx
	
	char f1[file1.length()];
	strcpy(f1,file1.c_str());
	
	char f2[file2.length()];
	strcpy(f2,file2.c_str()];
	
	std::ifstream aocx_stream(f1, std::ios::in|std::ios::binary);
        //checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionalNeuralNetwork.aocx");
        std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
        cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()+1));
	cl::Program program1(context1, DeviceList1, mybinaries);
	err = program1.build(DeviceList1);	


	std::ifstream aocx_stream2(f2, std::ios::in|std::ios::binary);
        //checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionNeuralNetwork.aocx");
        std::string prog2(std::istreambuf_iterator<char>(aocx_stream2), (std::istreambuf_iterator<char>()));
        cl::Program::Binaries mybinaries2 (1, std::make_pair(prog2.c_str(), prog2.length()+1));
	cl::Program program2(context2, DeviceList2, mybinaries);
	err = program2.build(DeviceList2);

	std::string file1_xml = GoogLeNet_DIR+"Inception"+std::to_string(2*rank)+".xml";
	std::string file2_xml = GoogLeNet_DIR+"Inception"+std::to_string(2*rank+1)+".xml";
	
	char f1_xml[file1.length()];
	strcpy(f1,file1_xml.c_str());
	std::vector<std::string> first_kernels;   //kernels from first aocx
	xml_parser(f1_xml,first_kernels);
	
	char f2_xml[file1.length()];
	strcpy(f2_xml,file2_xml.c_str());
	std::vector<std::string> second_kernels;   //kernels from second aocx
	xml_parser(f2_xml,second_kernels);
	

	cl::Kernel *kernels[52];
	int kernel_index = 0;
	//cl::CommandQueue *queues[50];
	cl::Buffer *buffers[100];
	int buffer_index = 0;
	//std::cout<<"Images array size: "<<sizeof(images)/sizeof(images[0])<<"\n";

	//Input image buffer is always mapped to 1st device context(Device ID =0)
	buffers[buffer_index] = new cl::Buffer(*contexts[0], CL_MEM_READ_ONLY, sizeof(cl_uchar) * dim_x * dim_y * num_images);
	err = cmd_queues[0]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_uchar) * dim_x * dim_y * num_images, images); //images buffer
	assert(err == CL_SUCCESS);
	err = cmd_queues[0]->finish();
	std::cout << " Error code after image transfer:" << kernel_index << " is ===>" << err << std::endl;
	assert(err == CL_SUCCESS);
	std::cout << "images copied\n";
	int num_filters = 0;
	int num_classes = 0;
	int num_pixels = dim_x * dim_y;

	// Launching the kernels, the first one with images as input.

	if (root == NULL)
	{
		std::cout << "Tree construction error\n";
		return -1;
	}
	else
	{ 
		std::queue<struct layersDetails *> q;

		q.push(root);

		while (!q.empty())
		{
			int n = q.size();

			while (n > 0)
			{

				struct layersDetails *p = q.front();
				q.pop();
				if (p != NULL)
				{
					//code to launch kernels
					if (p->layerType == "Convolution")
					{
						// To check on which device this  layer is mapped to.
						int layerDeviceNumber = p->device_num;
						kernels[kernel_index] = new cl::Kernel(*programs[layerDeviceNumber], "ConvolutionLayer", &err);
						assert(err == CL_SUCCESS);

						err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //first argument, input, also the output of the previous layer
						assert(err == CL_SUCCESS);
						buffer_index++;
						std::cout << "images passed\n";

						printFilterWeights(p->layerWeights);
						buffers[buffer_index] = new cl::Buffer(*contexts[p->layerID], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
						err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
						err = cmd_queues[p->layerID]->finish();
						std::cout << " Error code after weights transfer:" << kernel_index << " is ===>" << err << std::endl;
						assert(err == CL_SUCCESS);
						err = kernels[kernel_index]->setArg(1, *buffers[buffer_index]);
						assert(err == CL_SUCCESS);
						buffer_index++;

						std::cout << "weights passed\n";

						buffers[buffer_index] = new cl::Buffer(*contexts[p->layerID], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
						err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
						err = cmd_queues[p->layerID]->finish();
						std::cout << " Error code after bias transfer:" << kernel_index << " is ===>" << err << std::endl;
						assert(err == CL_SUCCESS);
						err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]);
						assert(err == CL_SUCCESS);
						buffer_index++;
						std::cout << "biases passed\n";

						int f_dim = p->params["kernel"].at(0) - '0';
						std::cout << "num of filter rows : " << f_dim << "\n";
						err = kernels[kernel_index]->setArg(3, f_dim); //filter rows
						assert(err == CL_SUCCESS);
						std::cout << "filter rows passed\n";

						err = kernels[kernel_index]->setArg(4, (int)f_dim); //filter cols
						assert(err == CL_SUCCESS);
						std::cout << "filter cols passed\n";

						num_filters = std::atoi(p->params["output"].c_str());
						std::cout << "no of filters from the map: " << num_filters << "\n";
						err = kernels[kernel_index]->setArg(5, num_filters); //no of filters
						assert(err == CL_SUCCESS);
						std::cout << "no of filters passed\n";

						err = kernels[kernel_index]->setArg(6, num_images); //no of images
						assert(err == CL_SUCCESS);
						std::cout << "no of images passed\n";

						err = kernels[kernel_index]->setArg(7, dim_x); //no of image rows
						assert(err == CL_SUCCESS);
						std::cout << "no of image rows passed\n";

						err = kernels[kernel_index]->setArg(8, dim_y); //no of image cols
						assert(err == CL_SUCCESS);
						std::cout << "no of image cols passed\n";

						int pad_begin[] = {p->params["pads_begin"].at(0) - '0', p->params["pads_begin"].at(2) - '0'};
						buffers[buffer_index] = new cl::Buffer(*contexts[p->layerID], CL_MEM_READ_ONLY, sizeof(cl_int) * 2);
						err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_int) * 2, pad_begin); //pad begin
						assert(err == CL_SUCCESS);
						err = cmd_queues[p->layerID]->finish();
						assert(err == CL_SUCCESS);
						//std::cout<<"padding from the map"<<pad<<"\n";
						err = kernels[kernel_index]->setArg(9, *buffers[buffer_index]); //padding
						assert(err == CL_SUCCESS);
						std::cout << "padding passed\n";
						buffer_index++;

						int pad_end[] = {p->params["pads_end"].at(0) - '0', p->params["pads_end"].at(2) - '0'};
						buffers[buffer_index] = new cl::Buffer(*contexts[p->layerID], CL_MEM_READ_ONLY, sizeof(cl_int) * 2);
						err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_int) * 2, pad_end); //pad end
						cmd_queues[p->layerID]->finish();
						//std::cout<<"padding from the map"<<pad<<"\n";
						err = kernels[kernel_index]->setArg(10, *buffers[buffer_index]); //padding
						assert(err == CL_SUCCESS);
						std::cout << "padding passed\n";
						buffer_index++;

						int stride = p->params["strides"].at(0) - '0';
						std::cout << "stride from the map(conv): " << stride << "\n";
						err = kernels[kernel_index]->setArg(11, stride); //stride
						assert(err == CL_SUCCESS);
						std::cout << "stride passed\n";
						//err = kernels[kernel_index]->setArg(9,dim_x);		//conv output rows
						//assert(err==CL_SUCCESS);

						//err = kernels[kernel_index]->setArg(10,dim_y);		//conv output cols
						//assert(err==CL_SUCCESS);

						/*
						buffers[buffer_index] = new cl::Buffer(mycontext, CL_MEM_READ_WRITE, sizeof(cl_double) * dim_x * dim_y * num_images * num_filters);
						err = kernels[kernel_index]->setArg(12, *buffers[buffer_index]); //output of conv
						assert(err == CL_SUCCESS);
						std::cout << "conv output passed\n";
						*/
						err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
						//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
						assert(err == CL_SUCCESS);
						std::cout << " Error code soon after conv layer for kernel:" << kernel_index << " is ===>" << err << std::endl;
						kernel_index++;

						//err = myqueue.finish();
						assert(err == CL_SUCCESS);

						std::cout << " Error code  after conv layer finish for kernel:" << kernel_index << " is ===>" << err << std::endl;

						num_pixels *= num_filters;
					}
					else if (p->layerType == "Pooling")
					{
						// To check on which device this  layer is mapped to.
						int layerPoolDeviceNumber = p->device_num;
						kernels[kernel_index] = new cl::Kernel(*programs[layerPoolDeviceNumber], "MaxPool", &err);
						assert(err == CL_SUCCESS);
						/*
						err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //first argument, input, also the output of the previous layer
						assert(err == CL_SUCCESS);
						buffer_index++;
						std::cout << "images passed\n";
						*/
						err = kernels[kernel_index]->setArg(0, dim_x); //conv output rows
						assert(err == CL_SUCCESS);
						std::cout << "conv output rows passed\n";
						err = kernels[kernel_index]->setArg(1, dim_y); //conv output cols
						assert(err == CL_SUCCESS);
						std::cout << "conv output cols passed\n";
						err = kernels[kernel_index]->setArg(2, num_filters); //no of filters
						assert(err == CL_SUCCESS);
						std::cout << "no of filters passed\n";
						int stride = p->params["strides"].at(0) - '0';
						std::cout << "Stride from the map: " << stride << "\n";
						err = kernels[kernel_index]->setArg(3, stride); //stride
						assert(err == CL_SUCCESS);
						std::cout << "stride passed: " << stride << "\n";
						err = kernels[kernel_index]->setArg(4, num_images); //no of images
						assert(err == CL_SUCCESS);
						std::cout << "no of images passed\n";
						//err = kernels[kernel_index]->setArg(6,dim_y);		//no of image cols
						//assert(err==CL_SUCCESS);
						dim_x /= stride;
						dim_y /= stride;
						num_pixels /= (stride * stride);
						std::cout << "dim x: " << dim_x << " dim y: " << dim_y << " num_pixels: " << num_pixels << "\n";

						/*
						buffers[buffer_index] = new cl::Buffer(mycontext, CL_MEM_READ_WRITE, sizeof(cl_double) * dim_x * dim_y * num_images * num_filters);
						err = kernels[kernel_index]->setArg(6, *buffers[buffer_index]); //output of pool
						assert(err == CL_SUCCESS);
						std::cout << "pool output\n";
						*/
						err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
						//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
						assert(err == CL_SUCCESS);
						kernel_index++;
					}
					else if (p->layerType == "FullyConnected")
					{
						// To check on which device this  layer is mapped to.
						int layerFCDeviceNumber = p->device_num;
						kernels[kernel_index] = new cl::Kernel(*programs[layerFCDeviceNumber], "FCL_Kernel", &err);
						assert(err == CL_SUCCESS);
						/*
						err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //first argument, input, also the output of the previous layer
						assert(err == CL_SUCCESS);
						buffer_index++;
						std::cout << "images passed\n";
						*/

						buffers[buffer_index] = new cl::Buffer(*contexts[p->layerID], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
						err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
						cmd_queues[p->layerID]->finish();
						assert(err == CL_SUCCESS);
						err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]);
						assert(err == CL_SUCCESS);
						buffer_index++;
						std::cout << "weights passed\n";
						buffers[buffer_index] = new cl::Buffer(*contexts[p->layerID], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
						err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
						cmd_queues[p->layerID]->finish();
						assert(err == CL_SUCCESS);
						err = kernels[kernel_index]->setArg(1, *buffers[buffer_index]);
						assert(err == CL_SUCCESS);
						buffer_index++;
						std::cout << "biases passed\n";
						err = kernels[kernel_index]->setArg(2, num_pixels); //no of pixels
						assert(err == CL_SUCCESS);
						std::cout << "no of pixels passed\n";
						num_classes = std::atoi(p->params["out-size"].c_str());
						std::cout << "num classes from the map: " << num_classes << "\n";
						err = kernels[kernel_index]->setArg(3, num_classes); //no of classes
						assert(err == CL_SUCCESS);
						std::cout << "no of classes passed\n";
						err = kernels[kernel_index]->setArg(4, num_images); //no of images
						assert(err == CL_SUCCESS);
						std::cout << "no of images passed\n";
						buffers[buffer_index] = new cl::Buffer(*contexts[p->layerID], CL_MEM_READ_WRITE, sizeof(cl_int) * num_images);
						err = kernels[kernel_index]->setArg(5, *buffers[buffer_index]); //output of FC
						assert(err == CL_SUCCESS);
						std::cout << "output of FC passed\n";
						err = kernels[kernel_index]->setArg(6, dim_x); //rows
						assert(err == CL_SUCCESS);
						std::cout << "no of rows passed\n";
						err = kernels[kernel_index]->setArg(7, dim_y); //cols
						assert(err == CL_SUCCESS);
						std::cout << "no of cols passed\n";
						err = kernels[kernel_index]->setArg(8, num_filters); //no of filters
						assert(err == CL_SUCCESS);
						std::cout << "no of filters passed\n";
						err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
						assert(err == CL_SUCCESS);
						kernel_index++;
					}
				}
				// Enqueue all children of the dequeued item
				for (std::vector<struct layersDetails *>::iterator it = p->children.begin(); it != p->children.end(); ++it)
				{
					if (*it != NULL)
						q.push(*it);
				}
				std::cout << "size of queue: " << q.size() << "\n";
				n--;
			}
		}
	}

	int final_labels[num_images];
	//Output CMd Queue from last device.
	err = cmd_queues[DeviceList_Master.size()-1]->enqueueReadBuffer(*buffers[buffer_index], CL_TRUE, 0, sizeof(cl_int) * num_images, final_labels);
	assert(err == CL_SUCCESS);

	err = cmd_queues[DeviceList_Master.size()-1]->finish();
	assert(err == CL_SUCCESS);

	for (int i = 0; i < num_images; i++)
	{
		std::cout << "Image " << imageNames[i] << " : " << final_labels[i] << "\n";
	}

	return 0;
}
