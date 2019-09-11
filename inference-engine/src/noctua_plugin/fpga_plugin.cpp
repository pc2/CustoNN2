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
#include <ctime>
#include <thread>
#include <assert.h>
#include <ie_builders.hpp>
#include <queue>
#include <sys/types.h>
#include <dirent.h>
#include <mpi.h>
#include <map>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>

// Directories where the aocx files will be stored

using namespace InferenceEngine;
using namespace boost;
using namespace boost::property_tree;

//List of layer types supported by the plugin
std::string supported_layers[7] = {"Convolution", "Pooling", "FullyConnected", "Concat", "Reshape", "Eltwise", "ScaleShift"};
// Layer Name - Layer ID Hashmap
std::map<std::string, int> layerIDMap;
//Vector to hold Layer IDs
std::vector<int> ID_list;

std::string G_BITSTREAM_DIR = "";

unsigned char *images;
int num_images, dim_x, dim_y, dim_depth;

/**
 * Data structure to store information of each layer
 */
struct layersDetails
{
	//Layer ID
	int layerID;
	// Layer Name
	std::string layerName;
	//Type of Layer
	std::string layerType;
	//Pointer to the Layer Bias vector
	float *layerBias;
	//Pointer to the Layer Weights vector
	float *layerWeights;
	//Total number of biases
	int num_biases;
	//Total number of weights
	int num_weights;

	//Hashmap for parameters ( kernel, padding,dilation,precision) with its values.
	std::map<std::string, std::string> params;
	//Vector of parent layers
	std::vector<std::string> inputLayerNames;
	//Vector of child layers
	std::vector<std::string> outputLayerNames;
	//Vector of pointers to the Layers Details Structure of Children nodes
	std::vector<struct layersDetails *> children;
	//Vector of pointers to the Layers Details Structure of parent nodes
	std::vector<struct layersDetails *> parents;

	int dummy = 0;
	// Buffer index of the output
	int layerOutBufferIndex = 0;
	// vector of buffer index
	std::vector<int> parentOutBufferIndex;
	// Output dimension
	int outH = 0, outW = 0, outDepth = 0;
	int visited = 0;
};

/**
 *  function to check if the layer is supported by the plugin.
 */
bool isLayerSupported(std::string layer_name)
{
	for (int i = 0; i < (sizeof(supported_layers) / sizeof(*supported_layers)); i++)
	{
		if (layer_name.compare(supported_layers[i]) == 0)
			return true;
	}

	return false;
}

/**
 *  function to check if the layer is already added in the Tree.
 */
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
 *  Rename the layer name to match it with kernels. 
 *  Here we replace '/' from layer name in the IR to '_'
 *  Remove "InceptionV1/InceptionV1/" from the name
 */
std::string rename_node_name(std::string strToSplit, char delimiter, std::string model_name)
{
	std::string nodeName = "";
	std::stringstream ss(strToSplit);
	std::string item;
	std::vector<std::string> splittedStrings;
	while (std::getline(ss, item, delimiter))
	{
		splittedStrings.push_back(item);
	}
	size_t len = splittedStrings.size();
	if (model_name == "googlenet")
	{
		//Remove the 1st "InceptionV1"
		for (size_t pos = 2; pos != len; pos++)
		{
			nodeName += splittedStrings[pos] + "_";
		}
	}
	else if (model_name == "resnet" || model_name == "resnet16")
	{
		for (std::size_t i = 0; i < splittedStrings.size(); ++i)
		{
			if (splittedStrings[i] == "bottleneck")
			{
				splittedStrings[i] = "bt";
			}
			//std::cout << splittedStrings[i] << '\n';
		}
		if (*splittedStrings.begin() == "resnet")
		{
			for (size_t pos = 3; pos != len; pos++)
			{
				nodeName += splittedStrings[pos] + "_";
			}
		}
		else
		{
			for (size_t pos = 0; pos != len; pos++)
			{
				nodeName += splittedStrings[pos] + "_";
			}
		}
	}
	//Remove the last "_"
	nodeName.pop_back();
	return nodeName;
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
 * Parse the input images using OpenCV
 */
void parse_images(std::vector<std::string> imageNames, InferenceEngine::CNNNetwork network)
{
	InputsDataMap inputInfo = network.getInputsInfo();

	if (inputInfo.size() != 1)
		throw std::logic_error("Sample supports topologies only with 1 input");

	auto inputInfoItem = *inputInfo.begin();
	inputInfoItem.second->setPrecision(Precision::U8);
	inputInfoItem.second->setLayout(Layout::NCHW);
	//inputInfoItem.second->setLayout(Layout::NHWC);
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
		dim_depth = inputInfoItem.second->getTensorDesc().getDims()[1];
		size_t image_size = inputInfoItem.second->getTensorDesc().getDims()[3] * inputInfoItem.second->getTensorDesc().getDims()[2];

		if (imagesData.empty())
			throw std::logic_error("Valid input images were not found!");
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
	//printImage(images, num_images, dim_x, dim_y);
}

/**
 * Tree Construction logic:
 */
struct layersDetails *parse_root(InferenceEngine::CNNNetwork network,std::string model_name)
{
	struct layersDetails *root = new layersDetails;
	details::CNNNetworkIterator it(network.actual);
	int no_of_layers = static_cast<int>(network.layerCount());

	//Parse CNNNetwork to Network to get the IDs
	//Get the Layers in the network and store the weights and bias in a structure for each layer.
	//std::vector<layersDetails> cnnLayersList;
	while (it != details::CNNNetworkIterator())
	{
		//std::cout<<
		CNNLayer::Ptr layer = *it;
		if (layer->name != "Mul1_/Fused_Mul_/FusedScaleShift_")
		{

		if (isLayerSupported(layer->type) )
		{
			root->layerName = layer->name;
			if (root->layerName == "Mul1")
			{
				std::replace(root->layerName.begin(), root->layerName.end(), '/', '_');
			}
			else
			{
				std::replace(root->layerName.begin(), root->layerName.end(), '/', '_');
				root->layerName = rename_node_name(root->layerName, '_',model_name);
			}

			root->layerType = layer->type;
			root->params = layer->params;
			root->num_biases = 0;
			root->num_weights = 0;
			root->outH = static_cast<int>(layer->outData[0]->getTensorDesc().getDims()[3]);
			root->outW = static_cast<int>(layer->outData[0]->getTensorDesc().getDims()[2]);
			root->outDepth = static_cast<int>(layer->outData[0]->getTensorDesc().getDims()[1]);
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
		}
		it++;
	}

	return NULL;
}
/**
 * Find a layer by its ID
 */
void findbyID(struct layersDetails *root, int id, struct layersDetails *parent)
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
				if (p->layerID == id && p->layerName != "dummy")
					p->parents.push_back(parent);
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
/**
 * function to remove dummy layers from the tree.
 */
void remove_dummy_child(struct layersDetails *node)
{
	for (int i = 0; i < node->children.size(); i++)
	{
		if (node->children.at(i)->layerName == "dummy")
		{
			node->children.erase(node->children.begin() + i);
		}
	}
}

/**
 * Level order traversal to find node with particular ID
 */
void find_missing_duplicates(struct layersDetails *root)
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

				if (p->layerName == "dummy")
				{
					findbyID(root, p->layerID, p->parents.at(0));
				}
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

void find_by_name(struct layersDetails *root, std::string layer_name, int buffer_index)
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
				if (p->layerName == layer_name)
				{
					for (struct layersDetails *ch : p->children)
					{
						ch->parentOutBufferIndex.push_back(buffer_index);
					}
				}
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

/**
 * TREE construction logic:
 */
struct layersDetails *parse_child(InferenceEngine::CNNNetwork network, std::string layer_name, struct layersDetails *root, struct layersDetails *parent, std::string model)
{
	const char *l_name = layer_name.c_str();
	//std::cout<<" Layer in parse_child:"<<layer_name<<std::endl;
	CNNLayerPtr layer = network.getLayerByName(l_name);

	if (isLayerSupported(layer->type))
	{
		auto search = layerIDMap.find(layer_name);
		int ID = search->second;
		//std::cout<<"Inside Layer: "<<layer_name<<"\n";
		//std::cout<<"Duplicate ? "<<isDuplicate(ID)<<"\n";
		if (isDuplicate(ID))
		{
			struct layersDetails *child = new layersDetails;
			child->layerID = ID;
			child->layerName = "dummy";
			child->dummy = 1;
			child->parents.push_back(parent);
			return child;
		}
		else
		{

			struct layersDetails *child = new layersDetails;
			//retrieving the child information
			child->layerName = layer->name;
			if (model == "resnet" && child->layerName == "Mul1")
			{
				std::replace(child->layerName.begin(), child->layerName.end(), '/', '_');
			}
			else
			{
				std::replace(child->layerName.begin(), child->layerName.end(), '/', '_');
				child->layerName = rename_node_name(child->layerName, '_',model);
			}
			child->layerType = layer->type;
			child->params = layer->params;
			child->num_biases = 0;
			child->num_weights = 0;
			child->outH = static_cast<int>(layer->outData[0]->getTensorDesc().getDims()[3]);
			child->outW = static_cast<int>(layer->outData[0]->getTensorDesc().getDims()[2]);
			child->outDepth = static_cast<int>(layer->outData[0]->getTensorDesc().getDims()[1]);

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

			child->parents.push_back(parent);
			for (std::vector<std::string>::iterator it = child->outputLayerNames.begin(); it != child->outputLayerNames.end(); ++it)
			{
				child->children.push_back(parse_child(network, *it, root, child,model));
			}
			return child;
		}
	}
	else
	{
		int outLayer = 0;
		struct layersDetails *temp;
		temp = new layersDetails;
		temp->layerName = "anyName";
		temp->layerType = "anypool";
		for (auto it : layer->outData[0]->getInputTo())
		{
			temp->outputLayerNames.push_back(it.second->name);
			outLayer++;
		}
		//std::cout<<"Unsupported Layer name: "<<layer->type<<" outputs vector size: "<<temp.outputLayerNames.size()<<std::endl;
		for (std::vector<std::string>::iterator it = temp->outputLayerNames.begin(); it != temp->outputLayerNames.end(); ++it)
		{
			parent->children.push_back(parse_child(network, *it, root, parent,model));
		}

		return temp;
	}
}
/**
 * Print the constructed tree
 */
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
				std::cout << p->layerID << " -- " << p->layerName << " -- " << p->layerType << std::endl;
				std::cout << "\t OutDim H:" << p->outH << " -- W:" << p->outW << " -- D:" << p->outDepth << std::endl;
				std::cout << "\t Num of childrens:" << p->children.size() << std::endl;
				std::cout << "\t Num of parents: " << p->parents.size() << std::endl;
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
/**
 * Print all the available Platforms
 */
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

/**
 * Print all the devices.
 */
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

const ptree &empty_ptree()
{
	static ptree t;
	return t;
}

std::vector<std::string> xml_parser1(char *filename)
{
	std::vector<std::string> kernel_names;
	ptree tree;
	read_xml(filename, tree);
	BOOST_FOREACH (ptree::value_type const &v, tree.get_child("board"))
	{
		if (v.first == "kernel")
		{
			kernel_names.push_back(v.second.get<std::string>("<xmlattr>.name"));
			// std::cout << kernel_names.back() << "\n";
		}
	}
	return kernel_names;
}

int get_program_num(std::string layerName, std::vector<std::string> first_kernels, std::vector<std::string> second_kernels)
{
	for (std::vector<std::string>::iterator it = first_kernels.begin(); it != first_kernels.end(); ++it)
	{
		if (layerName == *it)
		{
			return 1;
		}
	}

	for (std::vector<std::string>::iterator it = second_kernels.begin(); it != second_kernels.end(); ++it)
	{
		if (layerName == *it)
		{
			return 2;
		}
	}

	return 0;
}

/**
 * OPENVINO FPGA NOCTUA PLUGIN is implemented in this function
 */
std::vector<int> fpga_launcher(InferenceEngine::CNNNetwork network, char *model_path, std::vector<std::string> imageNames, std::string model_name, int rank, int TOP_N, std::string bitstream_dir, std::string opencl_design, std::string route_xml_path)
{
	std::cout << "In FPGA Launcher" << std::endl;
	std::cout << "Rank: " << rank << "\n";
	std::vector<int> classification_result;
	parse_images(imageNames, network);
	G_BITSTREAM_DIR = bitstream_dir;
	//build a map mapping routing(concat and feeder) kernels to channels
	std::map<std::string, unsigned int> route_map;
	if(opencl_design=="channel") {
		route_map = build_topo_map(route_xml_path);
	}
	cl_int err;

	std::vector<cl::Platform> PlatformList; //Platforms

	err = cl::Platform::get(&PlatformList);
	assert(err == CL_SUCCESS);
	std::cout << " Error code after Get Platform:"
			  << " is ===>" << err << std::endl;

	//printPlatforms(PlatformList);

	std::vector<cl::Device> DeviceList_Master, DeviceList1, DeviceList2;

	err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList_Master);
	assert(err == CL_SUCCESS);
	std::cout << " Error code after Get Device is ===> " << err << std::endl;

	//Adding the first device to a seperate List
	DeviceList1.push_back(DeviceList_Master[0]);

	DeviceList2.push_back(DeviceList_Master[1]);
	//Printing the Devices
	printDevices(DeviceList_Master);

	cl::Context *contexts[2];
	contexts[0] = new cl::Context(DeviceList1);
	contexts[1] = new cl::Context(DeviceList2);

	cl::CommandQueue *cmd_queues[250]; // To be dynamically allocated at kernel launch, one per kernel. the index  of cmd queue array is Layer ID.
	cl::CommandQueue *pad_queues[250];
	Builder::Network originalNetwork(network);

	// This map is to store LayerName <-> LayerID pair

	//Insert values the key-value pair
	for (const auto &layer : originalNetwork.getLayers())
	{
		//std:: cout<<layer.getId()<<" "<<layer.getName()<<" "<<layer.getType()<<" "<<layer.getParameters().size()<<std::endl;
		layerIDMap.insert(std::pair<std::string, int>(layer.getName(), layer.getId()));
	}
	std::cout << std::endl;

	struct layersDetails *root = parse_root(network,model_name);
	//root node obtained to which images will go as input

	//To obtain the rest of the tree structure
	//Iterating over the outputs of root node, i.e. it's children

	if (root != NULL)
	{
		for (std::vector<std::string>::iterator it = root->outputLayerNames.begin(); it != root->outputLayerNames.end(); ++it)
		{
			root->children.push_back(parse_child(network, *it, root, root,model_name));
		}
	}

	if (root == NULL)
	{
		std::cout << "Tree creation failed\n";
		exit(-1);
	}

	//std::cout << "Number of children of root: " << root->children.size() << "\n";
	find_missing_duplicates(root);

	printCNNTree(root);

	cl::Program *programs[2];
	std::string aocx_name = "";
	if (model_name == "googlenet")
	{
		aocx_name = "inception";
	}
	else if (model_name == "resnet")
	{
		aocx_name = "block";
	}
	else if (model_name == "resnet16")
	{
		aocx_name = "unit";
	}

	std::string file1 = G_BITSTREAM_DIR + aocx_name + std::to_string(2 * rank) + ".aocx";	 //first aocx
	std::string file2 = G_BITSTREAM_DIR + aocx_name + std::to_string(2 * rank + 1) + ".aocx"; //second aocx
	char f1[file1.length()];
	strcpy(f1, file1.c_str());

	char f2[file2.length()];
	strcpy(f2, file2.c_str());

	std::string file1_xml = G_BITSTREAM_DIR + aocx_name + std::to_string(2 * rank) + ".xml";
	std::string file2_xml = G_BITSTREAM_DIR + aocx_name + std::to_string(2 * rank + 1) + ".xml";
	std::vector<std::string> first_kernels, second_kernels;
	// std::cout << "path1: " << f1 << "\n";
	// std::cout << "path2: " << f2 << "\n";
	if(opencl_design=="global"){
		std::ifstream aocx_stream(f1, std::ios::in | std::ios::binary);
		//checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionalNeuralNetwork.aocx");
		std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
		cl::Program::Binaries mybinaries(1, std::make_pair(prog.c_str(), prog.length() + 1));
		programs[0] = new cl::Program(*contexts[0], DeviceList1, mybinaries);
		err = programs[0]->build(DeviceList1);
		std::cout <<rank <<" Error code after build 0" <<err << "\n";

		if (model_name == "googlenet" || model_name == "resnet16" || (model_name == "resnet" && (2 * rank + 1 != 5)))
		{
			std::ifstream aocx_stream2(f2, std::ios::in | std::ios::binary);
			//checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionNeuralNetwork.aocx");
			std::string prog2(std::istreambuf_iterator<char>(aocx_stream2), (std::istreambuf_iterator<char>()));
			cl::Program::Binaries mybinaries2(1, std::make_pair(prog2.c_str(), prog2.length() + 1));
			programs[1] = new cl::Program(*contexts[1], DeviceList2, mybinaries2);
			err = programs[1]->build(DeviceList2);
			std::cout <<rank <<" Error code after build 1" <<err << "\n";
		}

		char f1_xml[file1_xml.length()];
		strcpy(f1_xml, file1_xml.c_str());
		// std::cout << "xml_path1: " << f1_xml << "\n";
		first_kernels = xml_parser1(f1_xml); //kernels from first aocx

		if (model_name == "googlenet" || model_name == "resnet16" || (model_name == "resnet" && (2 * rank + 1 != 5)))
		{
			char f2_xml[file2_xml.length()];
			strcpy(f2_xml, file2_xml.c_str());
			// std::cout << "xml_path2: " << f2_xml << "\n";
			second_kernels = xml_parser1(f2_xml); //kernels from second aocx
		}
	}else if(opencl_design=="channel"){
		char f1_xml[file1_xml.length()];
		strcpy(f1_xml, file1_xml.c_str());

		char f2_xml[file2_xml.length()];
		strcpy(f2_xml, file2_xml.c_str());

		std::cout << rank << " Flashing aocx" << std::endl;
		if (rank % 2 == 0)
		{

			std::cout << rank << " Even Rank , 1st device - " << f1 << "  ,2nd device -" << f2 << std::endl;
			std::ifstream aocx_stream(f1, std::ios::in | std::ios::binary);
			//checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionalNeuralNetwork.aocx");
			std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
			cl::Program::Binaries mybinaries(1, std::make_pair(prog.c_str(), prog.length() + 1));
			programs[0] = new cl::Program(*contexts[0], DeviceList1, mybinaries);
			err = programs[0]->build(DeviceList1);

			std::ifstream aocx_stream2(f2, std::ios::in | std::ios::binary);
			//checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionNeuralNetwork.aocx");
			std::string prog2(std::istreambuf_iterator<char>(aocx_stream2), (std::istreambuf_iterator<char>()));
			cl::Program::Binaries mybinaries2(1, std::make_pair(prog2.c_str(), prog2.length() + 1));
			programs[1] = new cl::Program(*contexts[1], DeviceList2, mybinaries2);
			err = programs[1]->build(DeviceList2);

			first_kernels = xml_parser1(f1_xml);  //kernels from first aocx
			second_kernels = xml_parser1(f2_xml); //kernels from second aocx
		}
		else
		{

			std::cout << rank << " Odd Rank , 1st device - " << f2 << "  ,2nd device -" << f1 << std::endl;
			std::ifstream aocx_stream(f2, std::ios::in | std::ios::binary);
			//checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionalNeuralNetwork.aocx");
			std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
			cl::Program::Binaries mybinaries(1, std::make_pair(prog.c_str(), prog.length() + 1));
			programs[0] = new cl::Program(*contexts[0], DeviceList1, mybinaries);
			err = programs[0]->build(DeviceList1);

			std::ifstream aocx_stream2(f1, std::ios::in | std::ios::binary);
			//checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionNeuralNetwork.aocx");
			std::string prog2(std::istreambuf_iterator<char>(aocx_stream2), (std::istreambuf_iterator<char>()));
			cl::Program::Binaries mybinaries2(1, std::make_pair(prog2.c_str(), prog2.length() + 1));
			programs[1] = new cl::Program(*contexts[1], DeviceList2, mybinaries2);
			err = programs[1]->build(DeviceList2);

			first_kernels = xml_parser1(f2_xml);  //kernels from first aocx
			second_kernels = xml_parser1(f1_xml); //kernels from second aocx
		}
	}
	std::cout << "MPI_Barrier is here" << std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
	//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));

	std::cout << rank << " : Flashing aocx done" << std::endl;
	std::cout << rank << " : XMLs read" << std::endl;

	cl::Kernel *kernels[250];
	int kernel_index = 0;
	//cl::CommandQueue *queues[50];
	cl::Buffer *buffers[500];
	int buffer_index = 0;

	int com_sz;
	MPI_Comm_size(MPI_COMM_WORLD, &com_sz);

	float normalized_image[dim_x * dim_y * dim_depth * num_images];

	struct layersDetails *scaling_layer = new layersDetails;
	std::string scaling_layer_name = "Mul1_/Fused_Mul_/FusedScaleShift_";
	const char *l_name = scaling_layer_name.c_str();
	CNNLayerPtr layer = network.getLayerByName(l_name);
	scaling_layer->layerName = layer->name;
	scaling_layer->layerType = layer->type;
	for (auto const &x : layer->blobs)
	{
		if (x.first == "biases")
		{
			if (x.second->size() > 0)
			{
				scaling_layer->layerBias = new float[x.second->size()];
				scaling_layer->num_biases = x.second->size();
				for (int biasCount = 0; biasCount < x.second->size(); biasCount++)
				{
					scaling_layer->layerBias[biasCount] = x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[biasCount]; // put the layer bias in the list
				}
			}
		}
		else if (x.first == "weights")
		{
			if (x.second->size() > 0)
			{
				scaling_layer->layerWeights = new float[x.second->size()];
				scaling_layer->num_weights = x.second->size();
				for (int m = 0; m < x.second->size(); m++)
				{
					scaling_layer->layerWeights[m] = x.second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>()[m];
				}
			}
		}
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 224 * 224; j++)
		{
			normalized_image[(i * 224 * 224) + j] = float(images[((2 - i) * 224 * 224) + j]) * scaling_layer->layerWeights[i] + scaling_layer->layerBias[i];
		}
	}

	float transpose_image[224 * 224 * 3];
	for (int i = 0; i < 224 * 224; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			transpose_image[(i * 3) + j] = normalized_image[(j * 224 * 224) + i];
		}
	}
	std::cout << rank << ":Normalized the image" << std::endl;
	if (opencl_design=="global" && rank != 0)
	{
		char layer_name[50];
		MPI_Recv(layer_name, 50, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		std::string previous_l_name(layer_name);
		int dims_prev;
		MPI_Recv(&dims_prev, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		float prev_data[dims_prev];
		MPI_Recv(prev_data, dims_prev, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		buffers[buffer_index] = new cl::Buffer(*contexts[0], CL_MEM_READ_ONLY, sizeof(cl_float) * dims_prev);
		cmd_queues[0] = new cl::CommandQueue(*contexts[0], DeviceList1[0]);
		err = cmd_queues[0]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * dims_prev, prev_data); //images buffer
		assert(err == CL_SUCCESS);
		err = cmd_queues[0]->finish();
		find_by_name(root, previous_l_name, buffer_index);
		buffer_index++;
	}
	else
	{

		buffers[buffer_index] = new cl::Buffer(*contexts[0], CL_MEM_READ_ONLY, sizeof(cl_float) * dim_x * dim_y * dim_depth * num_images);
		if(opencl_design=="global"){
			cmd_queues[0] = new cl::CommandQueue(*contexts[0], DeviceList1[0]);
			err = cmd_queues[0]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * dim_x * dim_y * dim_depth * num_images, transpose_image); //images buffer
			assert(err == CL_SUCCESS);
			err = cmd_queues[0]->finish();
		}
		else{
			pad_queues[0] = new cl::CommandQueue(*contexts[0], DeviceList1[0]);
			err = pad_queues[0]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * dim_x * dim_y * dim_depth * num_images, transpose_image); //images buffer
			assert(err == CL_SUCCESS);
			err = pad_queues[0]->finish();
			assert(err == CL_SUCCESS);
		}
		
		buffer_index++;
		//std::cout << " Error code after image transfer:" << kernel_index << " is ===>" << err << std::endl;
		assert(err == CL_SUCCESS);
	}
	
	std::cout << rank << ":Launching kernels" << std::endl;

	int num_filters = 0;
	int num_classes = 0;
	int num_pixels = dim_x * dim_y * dim_depth * num_images;
	int padding_kernel_index = 0;
	int padding_out_index = 0;
	//Assing 0 as parent buffer index for root node
	root->parentOutBufferIndex.push_back(0);

	// Launching the kernels, the first one with images as input.
	if(opencl_design=="global"){
		classification_result = launcher_global(DeviceList1, DeviceList2, kernel_index, err, buffer_index,
											rank, com_sz, root, cmd_queues, contexts, kernels, programs,
											buffers, first_kernels, second_kernels, TOP_N, model_name);
	
	}else if(opencl_design=="channel"){
		classification_result = launcher_channel(kernel_index,
											  buffer_index, TOP_N, route_map, DeviceList1, DeviceList2,
											  root, kernels, programs, buffers, contexts, cmd_queues, pad_queues, rank, first_kernels, second_kernels);
	}											

	return classification_result;
}

std::vector<int> launcher_global(std::vector<cl::Device> DeviceList1,
								 std::vector<cl::Device> DeviceList2, int kernel_index, cl_int err,
								 int buffer_index, int rank, int com_sz, struct layersDetails *root,
								 cl::CommandQueue *cmd_queues[250], cl::Context *contexts[2],
								 cl::Kernel *kernels[250], cl::Program *programs[2],
								 cl::Buffer *buffers[500], std::vector<std::string> first_kernels, std::vector<std::string> second_kernels, int TOP_N, std::string model_name)
{

	std::vector<int> results;
	// Launching the kernels, the first one with images as input.
	if (root == NULL)
	{
		std::cout << "Tree construction error\n";
		exit(-1);
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
				while (p != NULL && p->layerName == "dummy")
				{
					p = q.front();
					q.pop();
					n--;
				}
				int program_number = get_program_num(p->layerName, first_kernels, second_kernels);
				if (p != NULL && program_number != 0)
				{
					const char *layerName = p->layerName.c_str();
					std::cout << "Launching Layer:" << layerName << std::endl;
					std::cout << get_program_num(p->layerName, first_kernels, second_kernels) << "\n";
					if (program_number == 1)
					{
						cmd_queues[p->layerID] = new cl::CommandQueue(*contexts[program_number - 1], DeviceList1[0]);
					}
					else
					{
						cmd_queues[p->layerID] = new cl::CommandQueue(*contexts[program_number - 1], DeviceList2[0]);
					}
					std::cout << "layer id = " << p->layerID << "\n";
					//code to launch kernels
					if (p->layerType == "Convolution" && program_number != 0)
					{
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1 || get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							std::cout << "\t Kernel Index:" << kernel_index << std::endl;
							std::cout << "\t Kernel Created " << std::endl;
							assert(err == CL_SUCCESS);
							std::cout << "\t  pads_begin :" << p->params["pads_begin"].at(0) << "," << p->params["pads_begin"].at(2) << " pads_end :"
									  << p->params["pads_begin"].at(0) << "," << p->params["pads_begin"].at(2) << std::endl;
							//For zero padding conv layer
							if (p->params["pads_begin"].at(0) == '0' && p->params["pads_begin"].at(2) == '0' && p->params["pads_end"].at(0) == '0' && p->params["pads_end"].at(2) == '0')
							{
								int pad_out_index = 0;
								//std::cout<<"\t Kernel Index:"<<kernel_index<<std::endl;
								kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
								assert(err == CL_SUCCESS);
								//output
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
								err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //output of conv
								assert(err == CL_SUCCESS);
								p->layerOutBufferIndex = buffer_index;
								for (struct layersDetails *ch : p->children)
								{
									ch->parentOutBufferIndex.push_back(
										p->layerOutBufferIndex);
								}
								buffer_index++;
								err = kernels[kernel_index]->setArg(1, *buffers[p->parentOutBufferIndex.at(0)]); //first argument, input, also the output of the previous layer
								assert(err == CL_SUCCESS);
								buffer_index++;
								//weights
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
								err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
								err = cmd_queues[p->layerID]->finish();
								//std::cout << "\t Error code after weights transfer:" << kernel_index << " is ===>" << err << std::endl;
								assert(err == CL_SUCCESS);
								err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]);
								assert(err == CL_SUCCESS);
								buffer_index++;
								//Bias
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
								err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
								err = cmd_queues[p->layerID]->finish();
								//std::cout << "\t Error code after bias transfer:" << kernel_index << " is ===>" << err << std::endl;
								assert(err == CL_SUCCESS);
								err = kernels[kernel_index]->setArg(3, *buffers[buffer_index]);
								assert(err == CL_SUCCESS);
								buffer_index++;
								//std::cout << "\tbiases passed\n";
								err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
								assert(err == CL_SUCCESS);
								kernel_index++;
								err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								p->visited = 1;
							}
							else
							{
								int inputIndex = 0;
								if (root->parents.size() == 0)
								{
									inputIndex = 0;
								}
								else
								{
									inputIndex = p->parentOutBufferIndex.at(0);
								}
								//Pad kernel launching code
								std::string pad_kernel_name = "Padding_" + p->layerName;
								if (model_name == "resnet" || model_name == "resnet16")
								{
									pad_kernel_name = "P_" + p->layerName;
								}
								const char *pad_name = pad_kernel_name.c_str();
								//std::cout<<"\t Kernel Index:"<<kernel_index<<std::endl;
								kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], pad_name, &err);
								assert(err == CL_SUCCESS);

								int pad_x = p->params["pads_begin"].at(0) - '0';
								int pad_y = p->params["pads_end"].at(0) - '0';
								int dim1 = 0;
								int dim2 = 0;
								int dim3 = 0;
								//Padding kernel output dimension calculation
								if ((model_name == "resnet" && p->layerName == "conv1_Conv2D") || (model_name == "resnet16" && p->layerName == "conv1_Conv2D") || (model_name == "googlenet" && p->layerName == "Conv2d_1a_7x7_Conv2D"))
								{
									//Set the dimensions for input layer
									dim1 = dim_x + pad_x + pad_y;
									dim2 = dim_y + pad_x + pad_y;
									dim3 = dim_depth;
								}
								else
								{
									//Output dimension of the parent layer
									dim1 = p->parents.at(0)->outH + pad_x + pad_y; // Add padding dimension
									dim2 = p->parents.at(0)->outW + pad_x + pad_y; // Add padding dimension
									dim3 = p->parents.at(0)->outDepth;			   //Depth remains same after padding
								}
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * dim1 * dim2 * dim3);
								int pad_out_index = buffer_index;
								err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]);
								assert(err == CL_SUCCESS);
								buffer_index++;
								err = kernels[kernel_index]->setArg(1, *buffers[p->parentOutBufferIndex.at(0)]); //input to pad
								err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
								assert(err == CL_SUCCESS);

								err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								kernel_index++;

								kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
								//std::cout << "\t Kernel Created "<<std::endl;
								assert(err == CL_SUCCESS);
								//output
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
								err = kernels[kernel_index]->setArg(0,
																	*buffers[buffer_index]); //output of conv
								assert(err == CL_SUCCESS);
								std::cout << "\tconv output passed :" << p->outH * p->outW * p->outDepth << std::endl;
								p->layerOutBufferIndex = buffer_index;
								if (p->children.size() > 0)
								{
									for (struct layersDetails *ch : p->children)
									{
										ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
									}
								}
								buffer_index++;
								//input
								err = kernels[kernel_index]->setArg(1, *buffers[pad_out_index]); //first argument, input, also the output of the previous layer

								assert(err == CL_SUCCESS);
								buffer_index++;

								//weights
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
								err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
								err = cmd_queues[p->layerID]->finish();
								//std::cout << "\t Error code after weights transfer:" << kernel_index << " is ===>" << err << std::endl;
								assert(err == CL_SUCCESS);
								err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]);
								assert(err == CL_SUCCESS);
								buffer_index++;
								//std::cout << "\tweights passed\n";
								//Bias
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
								err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
								err = cmd_queues[p->layerID]->finish();
								//std::cout << "\t Error code after bias transfer:" << kernel_index << " is ===>" << err << std::endl;
								assert(err == CL_SUCCESS);
								err = kernels[kernel_index]->setArg(3, *buffers[buffer_index]);
								assert(err == CL_SUCCESS);
								buffer_index++;

								err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);

								assert(err == CL_SUCCESS);

								err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								kernel_index++;
								p->visited = 1;
							}

							//Stop the execution after last conv
							if ((model_name == "googlenet" && p->layerName == "Conv2d_0c_1x1_Conv2D") || (model_name == "resnet" && p->layerName == "logits_Conv2D") || (model_name == "resnet16" && p->layerName == "logits_Conv2D"))
							{
								//Read buffer from the last conv output
								float convScores[p->outH * p->outW * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, convScores);
								err = cmd_queues[p->layerID]->finish();
								//Printing the results after getting Top N Results
								std::cout << " TOP- " << TOP_N << " Classification Scores" << std::endl;
								std::cout << " --------------------------------" << std::endl;
								results = getTopNResults(convScores, TOP_N);
								std::cout << " --------------------------------" << std::endl;
								//std::cout << " Please match the above labels with the \"Labels.txt\" of the model to see the classification results."<< std::endl;
								return results;
							}
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "Pooling" && program_number != 0)
					{
						// To check on which device this  layer is mapped to.
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1 || get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							std::cout << "\t kernel:" << layerName << std::endl;
							kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);
							//output
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
							err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //conv output rows
							assert(err == CL_SUCCESS);
							p->layerOutBufferIndex = buffer_index;
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							buffer_index++;
							//input
							err = kernels[kernel_index]->setArg(1, *buffers[p->parentOutBufferIndex.at(0)]); //no of images
							assert(err == CL_SUCCESS);
							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							assert(err == CL_SUCCESS);
							cmd_queues[p->layerID]->finish();
							assert(err == CL_SUCCESS);
							kernel_index++;
							p->visited = 1;
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "ScaleShift" && program_number != 0)
					{

						int flag_parents = 0;
						//std::cout<<"no of parents: "<<p->parents.size()<<"\n";
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1 || get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);
							//output
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
							err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //output of eltwise
							assert(err == CL_SUCCESS);
							p->layerOutBufferIndex = buffer_index;
							//std::cout<<"no of children : "<<p->children.size()<<"\n";
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							buffer_index++;
							//std::cout<<"parent out buffer index size: "<<p->parentOutBufferIndex.size()<<"\n";
							err = kernels[kernel_index]->setArg(1, *buffers[p->parentOutBufferIndex.at(0)]); //first argument, input, also the output of the previous layer
							assert(err == CL_SUCCESS);
							buffer_index++;
							//weights
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
							err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
							err = cmd_queues[p->layerID]->finish();
							//std::cout << "\t Error code after weights transfer:" << kernel_index << " is ===>" << err << std::endl;
							assert(err == CL_SUCCESS);
							err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]);
							assert(err == CL_SUCCESS);
							buffer_index++;
							//Bias
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
							err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
							err = cmd_queues[p->layerID]->finish();
							//std::cout << "\t Error code after bias transfer:" << kernel_index << " is ===>" << err << std::endl;
							assert(err == CL_SUCCESS);
							err = kernels[kernel_index]->setArg(3, *buffers[buffer_index]);
							assert(err == CL_SUCCESS);
							buffer_index++;
							//std::cout << "\tbiases passed\n";
							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							assert(err == CL_SUCCESS);
							kernel_index++;
							err = cmd_queues[p->layerID]->finish();
							assert(err == CL_SUCCESS);
							p->visited = 1;
						}
					}
					else if (p->layerType == "Eltwise" && program_number != 0)
					{

						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1 || get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
							{
								flag_parents++;
							}
						}
						//std::cout << "flag_parents: " << flag_parents << "\n";
						//std::cout << "p->parents.size(): " << p->parents.size() << "\n";
						if (!p->visited && flag_parents == p->parents.size())
						{
							//std::cout << "\t kernel:"<<layerName<<std::endl;
							kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);
							//output
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
							err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //elt output rows
							assert(err == CL_SUCCESS);
							p->layerOutBufferIndex = buffer_index;
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							buffer_index++;
							//input
							err = kernels[kernel_index]->setArg(1, *buffers[p->parents.at(0)->layerOutBufferIndex]); //elt input1
							assert(err == CL_SUCCESS);
							buffer_index++;
							err = kernels[kernel_index]->setArg(2, *buffers[p->parents.at(1)->layerOutBufferIndex]); //elt input2
							assert(err == CL_SUCCESS);
							buffer_index++;
							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
							assert(err == CL_SUCCESS);
							cmd_queues[p->layerID]->finish();
							assert(err == CL_SUCCESS);
							std::cout << "kernel is launched with error code"
									  << err << "\n";
							kernel_index++;
							p->visited = 1;
							if (program_number == 2 && (p->layerName == "block2_unit_4_bt_v2_add" || p->layerName == "block3_unit_6_bt_v2_add"))
							{
								std::string elt_layer_name = p->layerName;
								//std::cout << "we are here\n";
								MPI_Send(elt_layer_name.c_str(), elt_layer_name.size(), MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
								int dims1 = p->outH * p->outW * p->outDepth;
								MPI_Send(&dims1, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
								float elt_out[p->outH * p->outW * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, elt_out);
								MPI_Send(elt_out, p->outH * p->outW * p->outDepth, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
							}
							else if (model_name == "resnet16" && program_number == 2 && p->layerName != "block4_unit_3_bt_v2_add" )
							{
								std::string elt_layer_name = p->layerName;
								//std::cout << "we are here\n";
								MPI_Send(elt_layer_name.c_str(), elt_layer_name.size(), MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
								int dims1 = p->outH * p->outW * p->outDepth;
								MPI_Send(&dims1, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
								float elt_out[p->outH * p->outW * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, elt_out);
								MPI_Send(elt_out, p->outH * p->outW * p->outDepth, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
							}
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "FullyConnected" && program_number != 0)
					{
						// To check on which device this  layer is mapped to.
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
							err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
							cmd_queues[p->layerID]->finish();
							assert(err == CL_SUCCESS);
							err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]);
							assert(err == CL_SUCCESS);
							buffer_index++;
							std::cout << "\tweights passed\n";
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
							err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
							cmd_queues[p->layerID]->finish();
							assert(err == CL_SUCCESS);
							err = kernels[kernel_index]->setArg(1, *buffers[buffer_index]);
							assert(err == CL_SUCCESS);
							buffer_index++;
							std::cout << "\tbiases passed\n";
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_int) * num_images);
							err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]); //output of FC
							assert(err == CL_SUCCESS);
							std::cout << "\toutput of FC passed\n";
							p->layerOutBufferIndex = buffer_index;
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(
									p->layerOutBufferIndex);
							}
							buffer_index++;
							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
							assert(err == CL_SUCCESS);
							cmd_queues[p->layerID]->finish();
							kernel_index++;
							p->visited = 1;
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "Concat" && program_number != 0)
					{
						//std::cout << " \tLaunching concat\n";
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{

							kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);
							std::cout << "\tConcat Parents:" << p->parents.size() << std::endl;
							std::cout << "\toutput\n";
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
							err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //concat output rows
							assert(err == CL_SUCCESS);
							p->layerOutBufferIndex = buffer_index;
							buffer_index++;
							//std::cout << "\tconv1\n";
							//input
							err = kernels[kernel_index]->setArg(1, *buffers[p->parents.at(0)->layerOutBufferIndex]); //conv 1
							assert(err == CL_SUCCESS);
							buffer_index++;
							//std::cout << "\tconv2\n";
							err = kernels[kernel_index]->setArg(2, *buffers[p->parents.at(1)->layerOutBufferIndex]); //conv 2
							assert(err == CL_SUCCESS);
							buffer_index++;
							//std::cout << "\tconv3\n";
							err = kernels[kernel_index]->setArg(3, *buffers[p->parents.at(2)->layerOutBufferIndex]); //conv 3
							assert(err == CL_SUCCESS);
							buffer_index++;
							//std::cout << "\tconv4\n";
							err = kernels[kernel_index]->setArg(4, *buffers[p->parents.at(3)->layerOutBufferIndex]); //conv 4
							assert(err == CL_SUCCESS);
							buffer_index++;
							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							assert(err == CL_SUCCESS);
							cmd_queues[p->layerID]->finish();

							kernel_index++;
							p->visited = 1;
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							// MPI write to the next host instance
							if (rank < com_sz - 1 && program_number == 2)
							{
								std::string concat_layer_name = p->layerName;
								std::cout << "we are here\n";
								MPI_Send(concat_layer_name.c_str(), concat_layer_name.size(), MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
								int dims1 = p->outH * p->outW * p->outDepth;
								MPI_Send(&dims1, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
								float concat_out[p->outH * p->outW * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, concat_out);
								MPI_Send(concat_out, p->outH * p->outW * p->outDepth, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
							}
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "SoftMax" && program_number != 0)
					{
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							std::cout << "\t Input buffer index:" << p->parentOutBufferIndex.at(0) << std::endl;
							std::cout << " outH, OutW, outDepth:" << p->outH << "," << p->outW << "," << p->outDepth << std::endl;
							kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
							err = kernels[kernel_index]->setArg(1, *buffers[buffer_index]); //softmax output
							assert(err == CL_SUCCESS);
							p->layerOutBufferIndex = buffer_index;
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							buffer_index++;
							//input
							err = kernels[kernel_index]->setArg(0, *buffers[p->parentOutBufferIndex.at(0)]); //reshape input1
							assert(err == CL_SUCCESS);
							buffer_index++;
							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);

							assert(err == CL_SUCCESS);
							kernel_index++;
							p->visited = 1;
							cmd_queues[p->layerID]->finish();
							kernel_index++;
							std::cout << "\t Output buffer index:" << p->layerOutBufferIndex << std::endl;
							//Stop execution after softmax
							if (p->layerName == "Predictions_Softmax")
								exit(0);
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "Reshape" && program_number != 0)
					{
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);
							if (p->layerName == "Predictions_Reshape")
							{
								std::cout << "\t Input buffer index:" << p->parentOutBufferIndex.at(0) << std::endl;
								std::cout << " outH, OutW, outDepth:" << p->outH << "," << p->outW << "," << p->outDepth << std::endl;
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outDepth);
								err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //reshape output
								assert(err == CL_SUCCESS);
								p->layerOutBufferIndex = buffer_index;
								if (p->children.size() > 0)
								{
									for (struct layersDetails *ch : p->children)
									{
										ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
									}
								}
								buffer_index++;
								std::cout << "No of parents: " << p->parents.size() << "\n";
								//input
								err = kernels[kernel_index]->setArg(1, *buffers[p->parents.at(0)->layerOutBufferIndex]); //conv input1
								assert(err == CL_SUCCESS);
								buffer_index++;
								err = kernels[kernel_index]->setArg(2, *buffers[p->parents.at(1)->layerOutBufferIndex]); //conv input2
								assert(err == CL_SUCCESS);
								buffer_index++;
								err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
								assert(err == CL_SUCCESS);
								kernel_index++;
								p->visited = 1;
								cmd_queues[p->layerID]->finish();
								std::cout << "\t Output buffer index:" << p->layerOutBufferIndex << std::endl;
								float final_labels[p->outH * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outDepth, final_labels);
								std::cout << "\tLabels top 10\n";
								for (int i = 0; i < 10; i++)
									std::cout << final_labels[i] << "\n";
							}
							else
							{
								std::cout << "\t No supporting Reshape layer"
										  << std::endl;
							}
						}
						if (p->visited == 0)
							q.push(p);
					}
				}
				// Enqueue all children of the dequeued item
				if (p->visited == 1 || program_number == 0)
				{
					for (std::vector<struct layersDetails *>::iterator it = p->children.begin(); it != p->children.end(); ++it)
					{
						if (*it != NULL)
							q.push(*it);
					}
					//std::cout << "\tsize of queue: " << q.size() << "\n";
					n--;
				}
			}
		}
	}
	return results;
}

std::vector<int> launcher_channel(int kernel_index,
								  int buffer_index, int TOP_N, std::map<std::string, unsigned int> route_map,
								  std::vector<cl::Device> DeviceList1, std::vector<cl::Device> DeviceList2,
								  struct layersDetails *root, cl::Kernel *kernels[250],
								  cl::Program *programs[2], cl::Buffer *buffers[500],
								  cl::Context *contexts[2], cl::CommandQueue *cmd_queues[500],
								  cl::CommandQueue *pad_queues[500], int rank, std::vector<std::string> first_kernels, std::vector<std::string> second_kernels)
{
	std::cout << "In launcher_channel" << std::endl;
	std::vector<int> results;
	cl_int err;
	auto end = std::chrono::system_clock::now();
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	//Declaration of kernels and command queues for feeder kernels:
	cl::Kernel *kernel_feeder_3b, *kernel_feeder_3c, *kernel_feeder_4a, *kernel_feeder_4c, *kernel_feeder_4d, *kernel_feeder_4e, *kernel_feeder_4f, *kernel_feeder_5a, *kernel_feeder_5c;
	cl::CommandQueue *queuefeeder_3b, *queuefeeder_3c, *queuefeeder_4a, *queuefeeder_4b, *queuefeeder_4c, *queuefeeder_4d, *queuefeeder_4e, *queuefeeder_4f, *queuefeeder_5a, *queuefeeder_5c;
	unsigned int router_temp;
	std::cout << rank << "Launching feeder kernels" << std::endl;
	//Feeding kernels and Create Command queue for feeders for every node, there are different feeder kernels
	if (rank == 0)
	{
		//std::this_thread::sleep_for(std::chrono::milliseconds(500));
		kernel_feeder_3b = new cl::Kernel(*programs[1], "feeder_3b", &err);
		assert(err == CL_SUCCESS);

		queuefeeder_3b = new cl::CommandQueue(*contexts[1], DeviceList2[0]);
		assert(err == CL_SUCCESS);

		router_temp = route_map["3b_input"];
		kernel_feeder_3b->setArg(0, router_temp);
		err = queuefeeder_3b->enqueueTask(*kernel_feeder_3b);
		//std::this_thread::sleep_for(std::chrono::milliseconds(500));
	}
	else if (rank == 1)
	{

		kernel_feeder_3c = new cl::Kernel(*programs[1], "feeder_3c", &err);
		assert(err == CL_SUCCESS);
		queuefeeder_3c = new cl::CommandQueue(*contexts[1], DeviceList2[0]);
		assert(err == CL_SUCCESS);

		kernel_feeder_4a = new cl::Kernel(*programs[0], "feeder_4a", &err);
		assert(err == CL_SUCCESS);
		queuefeeder_4a = new cl::CommandQueue(*contexts[0], DeviceList1[0]);
		assert(err == CL_SUCCESS);

		router_temp = route_map["3c_input"];
		kernel_feeder_3c->setArg(0, router_temp);
		err = queuefeeder_3c->enqueueTask(*kernel_feeder_3c);
		//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));

		router_temp = route_map["4b_input"];
		kernel_feeder_4a->setArg(0, router_temp);
		err = queuefeeder_4a->enqueueTask(*kernel_feeder_4a);
		std::cout << rank << " is the rank. 4b_input is coming from " << router_temp << std::endl;
		//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
	}
	else if (rank == 2)
	{

		kernel_feeder_4c = new cl::Kernel(*programs[0], "feeder_4c", &err);
		assert(err == CL_SUCCESS);
		kernel_feeder_4d = new cl::Kernel(*programs[1], "feeder_4d", &err);
		assert(err == CL_SUCCESS);

		queuefeeder_4c = new cl::CommandQueue(*contexts[0], DeviceList1[0]);
		assert(err == CL_SUCCESS);
		queuefeeder_4d = new cl::CommandQueue(*contexts[1], DeviceList2[0]);
		assert(err == CL_SUCCESS);

		router_temp = route_map["4c_input"];
		kernel_feeder_4c->setArg(0, router_temp);
		err = queuefeeder_4c->enqueueTask(*kernel_feeder_4c);
		//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));

		router_temp = route_map["4d_input"];
		kernel_feeder_4d->setArg(0, router_temp);
		err = queuefeeder_4d->enqueueTask(*kernel_feeder_4d);
		//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
	}
	else if (rank == 3)
	{

		kernel_feeder_4e = new cl::Kernel(*programs[1], "feeder_4e", &err);
		assert(err == CL_SUCCESS);

		kernel_feeder_4f = new cl::Kernel(*programs[0], "feeder_4f", &err);
		assert(err == CL_SUCCESS);

		queuefeeder_4e = new cl::CommandQueue(*contexts[1], DeviceList2[0]);
		assert(err == CL_SUCCESS);
		queuefeeder_4f = new cl::CommandQueue(*contexts[0], DeviceList1[0]);
		assert(err == CL_SUCCESS);

		router_temp = route_map["4e_input"];
		kernel_feeder_4e->setArg(0, router_temp);
		std::cout << "4e_input is coming from " << route_map["4e_input"] << std::endl;
		err = queuefeeder_4e->enqueueTask(*kernel_feeder_4e);
		//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));

		router_temp = route_map["4f_input"];
		kernel_feeder_4f->setArg(0, router_temp);
		err = queuefeeder_4f->enqueueTask(*kernel_feeder_4f);
		//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
	}
	else
	{

		kernel_feeder_5a = new cl::Kernel(*programs[0], "feeder_5a", &err);
		assert(err == CL_SUCCESS);
		kernel_feeder_5c = new cl::Kernel(*programs[1], "feeder_5c", &err);
		assert(err == CL_SUCCESS);

		queuefeeder_5a = new cl::CommandQueue(*contexts[0], DeviceList1[0]);
		assert(err == CL_SUCCESS);
		queuefeeder_5c = new cl::CommandQueue(*contexts[1], DeviceList2[0]);
		assert(err == CL_SUCCESS);

		router_temp = route_map["5a_input"];
		kernel_feeder_5a->setArg(0, router_temp);
		err = queuefeeder_5a->enqueueTask(*kernel_feeder_5a);
		//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));

		router_temp = route_map["5c_input"];
		kernel_feeder_5c->setArg(0, router_temp);
		err = queuefeeder_5c->enqueueTask(*kernel_feeder_5c);
		//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
	}
	std::cout << rank << "Feeder kernels launched" << std::endl;
	// Launching the kernels, the first one with images as input.
	if (root == NULL)
	{
		std::cout << "Tree construction error\n";
		exit(-1);
	}
	else
	{
		std::cout << rank << "Launching from the tree" << std::endl;
		std::queue<struct layersDetails *> q;
		q.push(root);
		while (!q.empty())
		{
			int n = q.size();
			while (n > 0)
			{
				struct layersDetails *p = q.front();
				q.pop();

				while (p != NULL and p->layerName == "dummy")
				{
					std::cout << rank << "Q popped :" << p->layerName << std::endl;
					p = q.front();
					q.pop();
					n--;
				}
				//Get the program number
				int program_number = get_program_num(p->layerName, first_kernels, second_kernels);
				std::cout << rank << " Program number for layer:" << p->layerID << ":" << p->layerName << " On Device : " << program_number << std::endl;
				if (p != NULL and program_number != 0)
				{
					const char *layerName = p->layerName.c_str();
					std::cout << rank << " Launching Layer:" << p->layerID << ":" << layerName << " On Device : " << program_number - 1 << std::endl;
					std::cout << rank << " Program number :" << get_program_num(p->layerName, first_kernels, second_kernels) << std::endl;

					int flag_parents = 0;
					for (struct layersDetails *ch : p->parents)
					{
						if (ch->visited == 1 or get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
						{
							flag_parents++;
						}
					}
					if (!p->visited && flag_parents == p->parents.size())
					{
						if (program_number == 1)
						{
							cmd_queues[p->layerID] = new cl::CommandQueue(*contexts[program_number - 1], DeviceList1[0]);
							pad_queues[p->layerID] = new cl::CommandQueue(*contexts[program_number - 1], DeviceList1[0]);
						}
						else
						{
							cmd_queues[p->layerID] = new cl::CommandQueue(*contexts[program_number - 1], DeviceList2[0]);
							pad_queues[p->layerID] = new cl::CommandQueue(*contexts[program_number - 1], DeviceList2[0]);
						}
					}
					//code to launch kernels
					if (p->layerType == "Convolution" && program_number != 0)
					{
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1 or get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							//For zero padding conv layer
							if (p->params["pads_begin"].at(0) == '0' && p->params["pads_begin"].at(2) == '0' && p->params["pads_end"].at(0) == '0' && p->params["pads_end"].at(2) == '0')
							{
								int pad_out_index = 0;
								//std::cout<<"\t Kernel Index:"<<kernel_index<<std::endl;
								kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
								assert(err == CL_SUCCESS);

								//weights
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
								err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
								err = cmd_queues[p->layerID]->finish();
								//std::cout << "\t Error code after weights transfer:" << kernel_index << " is ===>" << err << std::endl;
								assert(err == CL_SUCCESS);
								err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]);
								assert(err == CL_SUCCESS);
								buffer_index++;
								//Bias
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
								err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
								err = cmd_queues[p->layerID]->finish();
								//std::cout << "\t Error code after bias transfer:" << kernel_index << " is ===>" << err << std::endl;
								assert(err == CL_SUCCESS);
								err = kernels[kernel_index]->setArg(1, *buffers[buffer_index]);
								assert(err == CL_SUCCESS);
								buffer_index++;
								//std::cout << "\tbiases passed\n";
								//Argument for getting convolution scores
								// Parents of concat kernels
								if (p->layerName == "Conv2d_0c_1x1_Conv2D" ||
									p->layerName == "Mixed_3b_Branch_0_Conv2d_0a_1x1_Conv2D" || p->layerName == "Mixed_3b_Branch_3_Conv2d_0b_1x1_Conv2D" ||
									p->layerName == "Mixed_3c_Branch_0_Conv2d_0a_1x1_Conv2D" || p->layerName == "Mixed_3c_Branch_3_Conv2d_0b_1x1_Conv2D" ||
									p->layerName == "Mixed_4b_Branch_0_Conv2d_0a_1x1_Conv2D" || p->layerName == "Mixed_4b_Branch_3_Conv2d_0b_1x1_Conv2D" ||
									p->layerName == "Mixed_4c_Branch_0_Conv2d_0a_1x1_Conv2D" || p->layerName == "Mixed_4c_Branch_3_Conv2d_0b_1x1_Conv2D" ||
									p->layerName == "Mixed_4d_Branch_0_Conv2d_0a_1x1_Conv2D" || p->layerName == "Mixed_4d_Branch_3_Conv2d_0b_1x1_Conv2D" ||
									p->layerName == "Mixed_4e_Branch_0_Conv2d_0a_1x1_Conv2D" || p->layerName == "Mixed_4e_Branch_3_Conv2d_0b_1x1_Conv2D" ||
									p->layerName == "Mixed_4f_Branch_0_Conv2d_0a_1x1_Conv2D" || p->layerName == "Mixed_4f_Branch_3_Conv2d_0b_1x1_Conv2D" ||
									p->layerName == "Mixed_5b_Branch_0_Conv2d_0a_1x1_Conv2D" || p->layerName == "Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D" ||
									p->layerName == "Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D" || p->layerName == "Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D")

								{

									std::cout << "Global memory transfer for :" << p->layerName << std::endl;
									buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
									err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]); //output of conv
									assert(err == CL_SUCCESS);
									p->layerOutBufferIndex = buffer_index;
									for (struct layersDetails *ch : p->children)
									{
										ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
									}

									if (p->layerName != "Conv2d_0c_1x1_Conv2D")
									{
										buffer_index++;
									}
								}

								err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
								//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));

								assert(err == CL_SUCCESS);
								kernel_index++;

								p->visited = 1;
							}
							else
							{
								std::cout << "Conv with padding" << std::endl;
								int inputIndex = 0;
								if (root->parents.size() == 0)
								{
									inputIndex = 0;
								}
								else
								{
									inputIndex = p->parentOutBufferIndex.at(0);
								}
								//Pad kernel launching code
								std::string pad_kernel_name = "Padding_" + p->layerName;
								const char *pad_name = pad_kernel_name.c_str();
								std::cout << "\t Kernel Index:" << kernel_index << std::endl;
								kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], pad_name, &err);
								assert(err == CL_SUCCESS);
								int pad_x = p->params["pads_begin"].at(0) - '0';
								int pad_y = p->params["pads_end"].at(0) - '0';
								int dim1 = 0, dim2 = 0, dim3 = 0;
								//Padding kernel output dimension calculation
								if (p->layerName == "Conv2d_1a_7x7_Conv2D")
								{
									//Set the dimensions for input layer
									dim1 = dim_x + pad_x + pad_y;
									dim2 = dim_y + pad_x + pad_y;
									dim3 = dim_depth;
								}
								else
								{
									//Output dimension of the parent layer
									dim1 = p->parents.at(0)->outH + pad_x + pad_y; // Add padding dimension
									dim2 = p->parents.at(0)->outW + pad_x + pad_y; // Add padding dimension
									dim3 = p->parents.at(0)->outDepth;			   //Depth remains same after padding
								}
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * dim1 * dim2 * dim3);
								int pad_out_index = buffer_index;
								//Argument only for 1st kernel.
								if (p->layerName == "Conv2d_1a_7x7_Conv2D")
								{
									std::cout << rank << " padding kernel argument" << std::endl;
									err = kernels[kernel_index]->setArg(0, *buffers[0]);
									assert(err == CL_SUCCESS);
								}

								std::cout << rank << " padding kernel launch" << std::endl;
								err = pad_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
								//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));

								assert(err == CL_SUCCESS);
								std::cout << rank << " padding kernel launched" << std::endl;

								kernel_index++;
								// Pad kernel launching code ends
								std::cout << rank << "\t Kernel Index:" << kernel_index << std::endl;
								kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
								std::cout << rank << "\t Kernel Created " << std::endl;
								assert(err == CL_SUCCESS);

								buffer_index++;

								//weights
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
								err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
								err = cmd_queues[p->layerID]->finish();
								std::cout << rank << "\t Error code after weights transfer:" << kernel_index << " is ===>" << err << std::endl;
								assert(err == CL_SUCCESS);
								err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]);
								assert(err == CL_SUCCESS);
								buffer_index++;

								//Bias
								buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
								err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
								err = cmd_queues[p->layerID]->finish();
								std::cout << rank << "\t Error code after bias transfer:" << kernel_index << " is ===>" << err << std::endl;
								assert(err == CL_SUCCESS);
								err = kernels[kernel_index]->setArg(1, *buffers[buffer_index]);
								assert(err == CL_SUCCESS);
								buffer_index++;

								//parents of concat kernels
								if (p->layerName == "Mixed_3b_Branch_1_Conv2d_0b_3x3_Conv2D" || p->layerName == "Mixed_3b_Branch_2_Conv2d_0b_3x3_Conv2D" ||
									p->layerName == "Mixed_3c_Branch_2_Conv2d_0b_3x3_Conv2D" || p->layerName == "Mixed_3c_Branch_1_Conv2d_0b_3x3_Conv2D" ||
									p->layerName == "Mixed_4b_Branch_2_Conv2d_0b_3x3_Conv2D" || p->layerName == "Mixed_4b_Branch_1_Conv2d_0b_3x3_Conv2D" ||
									p->layerName == "Mixed_4c_Branch_2_Conv2d_0b_3x3_Conv2D" || p->layerName == "Mixed_4c_Branch_1_Conv2d_0b_3x3_Conv2D" ||
									p->layerName == "Mixed_4d_Branch_2_Conv2d_0b_3x3_Conv2D" || p->layerName == "Mixed_4d_Branch_1_Conv2d_0b_3x3_Conv2D" ||
									p->layerName == "Mixed_4e_Branch_2_Conv2d_0b_3x3_Conv2D" || p->layerName == "Mixed_4e_Branch_1_Conv2d_0b_3x3_Conv2D" ||
									p->layerName == "Mixed_4f_Branch_2_Conv2d_0b_3x3_Conv2D" || p->layerName == "Mixed_4f_Branch_1_Conv2d_0b_3x3_Conv2D" ||
									p->layerName == "Mixed_5b_Branch_2_Conv2d_0a_3x3_Conv2D" || p->layerName == "Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D" ||
									p->layerName == "Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D" || p->layerName == "Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D")
								{
									std::cout << "Global memory transfer for :" << p->layerName << std::endl;
									buffers[buffer_index] = new cl::Buffer(*contexts[program_number - 1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
									err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]); //output of conv
									assert(err == CL_SUCCESS);
									p->layerOutBufferIndex = buffer_index;
									for (struct layersDetails *ch : p->children)
									{
										ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
									}
									buffer_index++;
								}

								std::cout << rank << " conv kernel launch" << std::endl;
								err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
								std::cout << rank << " conv kernel launched" << std::endl;
								//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
								//std::cout << rank << " Waited for " << waitTime << "ms" << std::endl;

								//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
								assert(err == CL_SUCCESS);
								std::cout << rank << "\t Error code soon after conv layer for kernel:" << p->layerName << " is ===>" << err << std::endl;
								//err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								kernel_index++;
								p->visited = 1;
							}
							std::cout << rank << "\t Error code  after conv layer finish for kernel:" << kernel_index << " is ===>" << err << std::endl;
							std::cout << rank << "\t Error code  after conv layer finish for kernel named " << p->layerName << " is ===>" << err << std::endl;

							//Stop the execution after last conv
							if (p->layerName == "Conv2d_0c_1x1_Conv2D")
							{
								err = cmd_queues[p->layerID]->finish();
								//Read buffer from the last conv output
								float convScores[p->outH * p->outW * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, convScores);
								err = cmd_queues[p->layerID]->finish();

								//Printing the results after getting Top N Results
								std::cout << " TOP- " << TOP_N << " Classification scores" << std::endl;
								std::cout << " --------------------------------" << std::endl;
								results = getTopNResults(convScores, TOP_N);
								std::cout << " --------------------------------" << std::endl;
								////std::cout << " Please match the above labels with the \"Labels.txt\" of the model to see the classification results." << std::endl;
								//Stop Condition.
								return results;
							}
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "Pooling" && program_number != 0)
					{
						// To check on which device this  layer is mapped to.
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1 or get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							//std::cout << "\t kernel:"<<layerName<<std::endl;
							kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);

							p->layerOutBufferIndex = buffer_index;
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							buffer_index++;
							unsigned int router_temp;
							if (p->layerName == "MaxPool_3a_3x3_MaxPool")
							{
								router_temp = route_map["3a_output"];
								kernels[kernel_index]->setArg(0, router_temp);
							}

							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
							std::cout << rank << "\t P layer ID is " << p->layerID << std::endl;
							std::cout << rank << "\t Error code  after Pooling layer kernel:" << kernel_index << " is ===>" << err << std::endl;
							assert(err == CL_SUCCESS);

							kernel_index++;
							p->visited = 1;
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "Concat" && program_number != 0)
					{
						//std::cout << " \tLaunching concat\n";
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1 or get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{

							kernels[kernel_index] = new cl::Kernel(*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);

							unsigned int router_temp;

							if (p->layerName == "Mixed_3b_concat")
							{
								router_temp = route_map["3b_output"];
								std::cout << "\t Check 3b_output is :" << router_temp << std::endl;
								kernels[kernel_index]->setArg(0, router_temp);
							}
							if (p->layerName == "Mixed_3c_concat")
							{
								router_temp = route_map["3c_output"];
								std::cout << rank << " is the rank. 3c_output concat routes to  " << router_temp << std::endl;

								kernels[kernel_index]->setArg(0, router_temp);
							}
							if (p->layerName == "Mixed_4b_concat")
							{
								router_temp = route_map["4b_output"];
								kernels[kernel_index]->setArg(0, router_temp);
							}
							if (p->layerName == "Mixed_4c_concat")
							{
								router_temp = route_map["4c_output"];
								kernels[kernel_index]->setArg(0, router_temp);
							}
							if (p->layerName == "Mixed_4d_concat")
							{
								router_temp = route_map["4d_output"];
								kernels[kernel_index]->setArg(0, router_temp);
							}
							if (p->layerName == "Mixed_4e_concat")
							{
								router_temp = route_map["4e_output"];
								kernels[kernel_index]->setArg(0, router_temp);
							}
							if (p->layerName == "Mixed_4f_concat")
							{
								router_temp = route_map["4f_output"];
								kernels[kernel_index]->setArg(0, router_temp);
							}
							if (p->layerName == "Mixed_5b_concat")
							{
								router_temp = route_map["5b_output"];
								kernels[kernel_index]->setArg(0, router_temp);
							}
							if (p->layerName == "Mixed_5c_concat")
							{
								err = kernels[kernel_index]->setArg(0, *buffers[p->parents.at(0)->layerOutBufferIndex]); //conv 1
								assert(err == CL_SUCCESS);
								buffer_index++;

								//std::cout << "\tconv2\n";
								err = kernels[kernel_index]->setArg(1, *buffers[p->parents.at(1)->layerOutBufferIndex]); //conv 2
								assert(err == CL_SUCCESS);
								buffer_index++;

								//std::cout << "\tconv3\n";
								err = kernels[kernel_index]->setArg(2, *buffers[p->parents.at(2)->layerOutBufferIndex]); //conv 3
								assert(err == CL_SUCCESS);
								buffer_index++;

								//std::cout << "\tconv4\n";
								err = kernels[kernel_index]->setArg(3, *buffers[p->parents.at(3)->layerOutBufferIndex]); //conv 4
								assert(err == CL_SUCCESS);
								buffer_index++;

								cmd_queues[p->parents.at(0)->layerID]->finish();
								cmd_queues[p->parents.at(1)->layerID]->finish();
								cmd_queues[p->parents.at(2)->layerID]->finish();
								cmd_queues[p->parents.at(3)->layerID]->finish();
								err = cmd_queues[p->parents.at(2)->layerID]->enqueueTask(*kernels[kernel_index]);
								std::cout << rank << " is the rank. " << p->layerName << " launched on parent : " << p->parents.at(0)->layerName << " of ID : " << p->parents.at(0)->layerID << std::endl;
								//	cmd_queues[p->parents.at(2)->layerID]->finish();
							}
							else
							{
								err = kernels[kernel_index]->setArg(1, *buffers[p->parents.at(0)->layerOutBufferIndex]); //conv 1
								assert(err == CL_SUCCESS);
								buffer_index++;

								//std::cout << "\tconv2\n";
								err = kernels[kernel_index]->setArg(2, *buffers[p->parents.at(1)->layerOutBufferIndex]); //conv 2
								assert(err == CL_SUCCESS);
								buffer_index++;

								//std::cout << "\tconv3\n";
								err = kernels[kernel_index]->setArg(3, *buffers[p->parents.at(2)->layerOutBufferIndex]); //conv 3
								assert(err == CL_SUCCESS);
								buffer_index++;

								//std::cout << "\tconv4\n";
								err = kernels[kernel_index]->setArg(4, *buffers[p->parents.at(3)->layerOutBufferIndex]); //conv 4
								assert(err == CL_SUCCESS);
								buffer_index++;
								cmd_queues[p->parents.at(0)->layerID]->finish();
								cmd_queues[p->parents.at(1)->layerID]->finish();
								cmd_queues[p->parents.at(2)->layerID]->finish();
								cmd_queues[p->parents.at(3)->layerID]->finish();
								err = cmd_queues[p->parents.at(2)->layerID]->enqueueTask(*kernels[kernel_index]);
								std::cout << rank << " is the rank. " << p->layerName << " launched on parent : " << p->parents.at(0)->layerName << " of ID : " << p->parents.at(0)->layerID << std::endl;
								 
							}

							//err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);//moved inside for launching on parent`s queue
							std::cout << rank << "\t Error code  after Concat layer kernel:" << kernel_index << " is ===>" << err << std::endl;
							std::cout << rank << "\t Error code  after Concat layer named :" << p->layerName << " is ===>" << err << std::endl;
							//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
							//std::cout << rank << "is the rank. \t waited for :" << waitTime << " milliseconds " << std::endl;
							assert(err == CL_SUCCESS);

							//cmd_queues[p->layerID]->finish();

							kernel_index++;
							p->visited = 1;
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							assert(err == CL_SUCCESS);
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "SoftMax" && program_number != 0)
					{
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1 or get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							std::cout << "\t Input buffer index:"
									  << p->parentOutBufferIndex.at(0)
									  << std::endl;
							std::cout << " outH, OutW, outDepth:" << p->outH
									  << "," << p->outW << "," << p->outDepth
									  << std::endl;
							kernels[kernel_index] = new cl::Kernel(
								*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);
							buffers[buffer_index] = new cl::Buffer(
								*contexts[program_number - 1], CL_MEM_READ_WRITE,
								sizeof(cl_float) * p->outDepth);
							err = kernels[kernel_index]->setArg(1,
																*buffers[buffer_index]); //softmax output
							assert(err == CL_SUCCESS);
							p->layerOutBufferIndex = buffer_index;
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(
									p->layerOutBufferIndex);
							}
							buffer_index++;
							//input
							err = kernels[kernel_index]->setArg(0,
																*buffers[p->parentOutBufferIndex.at(0)]); //reshape input1
							assert(err == CL_SUCCESS);
							buffer_index++;
							err = cmd_queues[p->layerID]->enqueueTask(
								*kernels[kernel_index]);
							//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
							//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
							assert(err == CL_SUCCESS);
							kernel_index++;
							p->visited = 1;
							cmd_queues[p->layerID]->finish();
							kernel_index++;
							std::cout << "\t Output buffer index:"
									  << p->layerOutBufferIndex << std::endl;
							//Stop execution after softmax
							if (p->layerName == "Predictions_Softmax")
								exit(0);
						}
						if (p->visited == 0)
							q.push(p);
					}
					else if (p->layerType == "Reshape" && program_number != 0)
					{
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if (ch->visited == 1 or get_program_num(ch->layerName, first_kernels, second_kernels) == 0)
							{
								flag_parents++;
							}
						}
						if (!p->visited && flag_parents == p->parents.size())
						{
							kernels[kernel_index] = new cl::Kernel(
								*programs[program_number - 1], layerName, &err);
							assert(err == CL_SUCCESS);
							if (p->layerName == "Predictions_Reshape")
							{
								std::cout << "\t Input buffer index:"
										  << p->parentOutBufferIndex.at(0)
										  << std::endl;
								std::cout << " outH, OutW, outDepth:" << p->outH
										  << "," << p->outW << "," << p->outDepth
										  << std::endl;
								buffers[buffer_index] = new cl::Buffer(
									*contexts[program_number - 1],
									CL_MEM_READ_WRITE,
									sizeof(cl_float) * p->outH * p->outDepth);
								err = kernels[kernel_index]->setArg(0,
																	*buffers[buffer_index]); //reshape output
								assert(err == CL_SUCCESS);
								p->layerOutBufferIndex = buffer_index;
								if (p->children.size() > 0)
								{
									for (struct layersDetails *ch : p->children)
									{
										ch->parentOutBufferIndex.push_back(
											p->layerOutBufferIndex);
									}
								}
								buffer_index++;
								std::cout << "No of parents: "
										  << p->parents.size() << std::endl;
								;
								//input
								err =
									kernels[kernel_index]->setArg(1,
																  *buffers[p->parents.at(0)->layerOutBufferIndex]); //conv input1
								assert(err == CL_SUCCESS);
								buffer_index++;
								err =
									kernels[kernel_index]->setArg(2,
																  *buffers[p->parents.at(1)->layerOutBufferIndex]); //conv input2
								assert(err == CL_SUCCESS);
								buffer_index++;
								err = cmd_queues[p->layerID]->enqueueTask(
									*kernels[kernel_index]);
								//std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
								//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
								assert(err == CL_SUCCESS);
								kernel_index++;
								p->visited = 1;
								cmd_queues[p->layerID]->finish();
								std::cout << "\t Output buffer index:"
										  << p->layerOutBufferIndex << std::endl;
								float final_labels[p->outH * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(
									*buffers[p->layerOutBufferIndex],
									CL_TRUE, 0,
									sizeof(cl_float) * p->outH * p->outDepth, final_labels);
								std::cout << "\tLabels top 10" << std::endl;
								for (int i = 0; i < 10; i++)
									std::cout << final_labels[i] << std::endl;
							}
							else
							{
								std::cout << "\t No supporting Reshape layer"
										  << std::endl;
							}
						}
						if (p->visited == 0)
							q.push(p);
					}
				}
				// Enqueue all children of the dequeued item
				if (p->visited == 1 || program_number == 0)
				{
					for (std::vector<struct layersDetails *>::iterator it =
							 p->children.begin();
						 it != p->children.end();
						 ++it)
					{
						if (*it != NULL)
							q.push(*it);
					}
					//std::cout << "\tsize of queue: " << q.size() << "\n";
					n--;
				}
			}
		}
	}

	return results;
}

std::vector<int> getTopNResults(float final_labels[], int topN)
{
	std::vector<int> results;
	std::multimap<float, int, std::greater<float>> sorted_map;
	int N = 0;
	float tensor = 0, tensor1 = 0;
	float tensor2[1001];
    for (int ax1 = 0; ax1 < 1001; ++ax1)
    {
        tensor = -3.402823e+38f;
        for (int k1 = 0; k1 < 1001; ++k1)
        {
            tensor = std::max(tensor, final_labels[k1]);
        }
        tensor1 = 0.000000e+00f;
        for (int k2 = 0; k2 < 1001; ++k2)
        {
            tensor1 = (tensor1 + std::exp((final_labels[k2] - tensor)));
        }
        tensor2[ax1] = (std::exp((final_labels[ax1] - tensor)) / tensor1);
    }

	for (int i = 0; i < 1001; i++)
	{
		//convScores.insert(i,final_labels[i]);
		sorted_map.insert(std::make_pair(final_labels[i], i));
	}
	for (auto entry : sorted_map)
	{
		if (N < topN)
		{
			results.push_back(entry.second);
			std::cout << "Label Number: " << entry.second << " - Score " << entry.first << " - Softmax (in % ) :"<<tensor2[entry.second]*100<< std::endl;
		}
		else
		{
			break;
		}
		N++;
	}
	return results;
}

/**
 * Topology of FPGAs
 * To be built using xml parser. This is just a dummy placeholder
 */
std::map<std::string, unsigned int> build_topo_map(std::string xmlpath)
{
	std::cout << "Parsing routing xml :" << xmlpath << std::endl;
	const char *filename = xmlpath.c_str();
	std::map<std::string, unsigned int> route_map;
	ptree tree;
	read_xml(filename, tree);
	BOOST_FOREACH (ptree::value_type const &v, tree.get_child("googlenet"))
	{
		if (v.first == "inception")
		{
			std::string module = v.second.get<std::string>("module");
			unsigned int input = v.second.get<unsigned int>("input");
			unsigned int output = v.second.get<unsigned int>("output");
			//Channel directions can be only between 0 - 3, else stop the execution
			if (input >= 4 || output >= 4 || input < 0 || output < 0)
			{
				std::cout << "ERROR : invalid input/output channel direction for the inception:" << module << ". Please check the routing XML" << std::endl;
				exit(-1);
			}
			std::cout << module + "_input"
					  << ":" << input << ", " << module + "_output"
					  << ":" << output << std::endl;
			route_map.insert(std::pair<std::string, int>(module + "_input", input));
			route_map.insert(std::pair<std::string, int>(module + "_output", output));
		}
	}
	std::cout << "Topology map has been created" << std::endl;
	return route_map;
}
