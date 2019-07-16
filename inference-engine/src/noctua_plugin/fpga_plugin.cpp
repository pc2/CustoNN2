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
#include "mpi.h"

// Directories where the aocx files will be stored


using namespace InferenceEngine;
using namespace boost;
using namespace boost::property_tree;

//List of layer types supported by the plugin
std::string supported_layers[6] = {"Convolution", "Pooling", "FullyConnected", "Concat","Reshape"};
// Layer Name - Layer ID Hashmap
std::map<std::string, int> layerIDMap;
//Vector to hold Layer IDs
std::vector<int> ID_list;

std::string GoogLeNet_DIR = "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/GoogLeNet";
std::string ResNet_DIR = "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/ResNet";

unsigned char *images;
int num_images, dim_x, dim_y,dim_depth;
// OUTPUT WRITE BEING //

//Set to 1 if you want the outputs of each layers to be written to a file.
int outputWriteFlag = 1;
// Results File Prefix
std:: string resultsFileAppender = "Results__";
// OUTPUT WRITE END //

// Top N labels classification 
int TOP_N = 10 ;


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
	for (int i = 0; i < 6; i++)
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
std::string rename_node_name(std::string strToSplit, char delimiter)
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
	//Remove the 1st "InceptionV1"
	for (size_t pos = 2; pos != len; pos++)
	{
		nodeName += splittedStrings[pos] + "_";
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
struct layersDetails *parse_root(InferenceEngine::CNNNetwork network)
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
		if (isLayerSupported(layer->type))
		{
			root->layerName = layer->name;
			std::replace(root->layerName.begin(), root->layerName.end(), '/', '_');
			root->layerName = rename_node_name(root->layerName, '_');
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
		it++;
	}

	return NULL;
}
/**
 * Find a layer by its ID
 */
void findbyID(struct layersDetails *root, int id, struct layersDetails *parent)
{
	std::queue <struct layersDetails *> q;


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
				if (p->layerID == id&&p->layerName!="dummy")
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
	for(int i=0;i<node->children.size();i++)
	{
		if(node->children.at(i)->layerName=="dummy")
		{
			node->children.erase(node->children.begin()+i);
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
				
				if(p->layerName=="dummy")
					{
						findbyID(root,p->layerID,p->parents.at(0));
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
struct layersDetails *parse_child(InferenceEngine::CNNNetwork network, std::string layer_name, struct layersDetails *root,struct layersDetails *parent)
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
			std::replace(child->layerName.begin(), child->layerName.end(), '/', '_');
			child->layerName = rename_node_name(child->layerName, '_');
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
				child->children.push_back(parse_child(network, *it, root, child));
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
		//std::cout<<"Unsupported Layer name: "<<layer->type<<" outputs vector size: "<<temp.outputLayerNames.size()<<std::endl;
		if (temp.outputLayerNames.size() > 0)
			return parse_child(network, temp.outputLayerNames.front(), root,parent);
		else
			return NULL;
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
				std::cout <<"\t OutDim H:"<< p->outH << " -- W:" << p->outW << " -- D:" << p->outDepth << std::endl;
				std::cout << "\t Num of childrens:" << p->children.size()<<std::endl;
				std::cout<<"\t Num of parents: "<<p->parents.size()<<std::endl;
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

const ptree& empty_ptree(){
    static ptree t;
    return t;
}

void xml_parser1(char *filename,std::vector<std::string> kernel_names)
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
			kernel_names.push_back(v.second.data());
		}
        }
    }
	

}

int get_program_num(std::string layerName,std::vector<std::string> first_kernels,std::vector<std::string> second_kernels)
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
int fpga_launcher(InferenceEngine::CNNNetwork network, char *model_path, std::vector<std::string> imageNames, std::string model_name,int rank)
{
	std::cout << "In FPGA Launcher" << std::endl;
	
	parse_images(imageNames, network);

	cl_int err;

	std::vector<cl::Platform> PlatformList; //Platforms

	err = cl::Platform::get(&PlatformList);
	assert(err == CL_SUCCESS);
	std::cout << " Error code after Get Platform:"
			  << " is ===>" << err << std::endl;

	//printPlatforms(PlatformList);


	std::vector<cl::Device> DeviceList_Master,DeviceList1,DeviceList2;

	//std::vector<cl::Device> DeviceList_Master, DeviceList1;

	//std::vector<std::vector<cl::Device>> Devices_to_flash; //Devices
	//Printing the Devices available for the given platform.
	err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList_Master);
	assert(err == CL_SUCCESS);
	std::cout << " Error code after Get Device is ===> " << err << std::endl;

	//Adding the first device to a seperate List
	DeviceList1.push_back(DeviceList_Master[0]);

	DeviceList2.push_back(DeviceList_Master[1]);
	//Printing the Devices


	
	
	cl::Context *contexts[2];
	contexts[0] = new cl::Context(DeviceList1);
	contexts[1] = new cl::Context(DeviceList2);
	//cl::Context context1(DeviceList1);
	//cl::Context context2(DeviceList2);
	//cl::Context mycontext(DeviceList1); //Context
	//cl::Context mycontext1(DeviceList2);
	cl::CommandQueue *cmd_queues[250]; // To be dynamically allocated at kernel launch, one per kernel. the index  of cmd queue array is Layer ID.
	


	



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
			root->children.push_back(parse_child(network, *it, root,root));
		}
	}

	std::cout << "Number of children of root: " << root->children.size() << "\n";
	find_missing_duplicates(root);

	//printCNNTree(root);


	// Code for assigning affinity to layers, device numbering starts at 0
	

	// Code for Affinity ends

	//printCNNTree(root);
	
	cl::Program *programs[2];

	std::string file1 = GoogLeNet_DIR+"Inception"+std::to_string(2*rank)+".aocx";     //first aocx
	std::string file2 = GoogLeNet_DIR+"Inception"+std::to_string(2*rank+1)+".aocx";   //second aocx
	
	char f1[file1.length()];
	strcpy(f1,file1.c_str());
	
	char f2[file2.length()];
	strcpy(f2,file2.c_str());
	
	std::ifstream aocx_stream(f1, std::ios::in|std::ios::binary);
        //checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionalNeuralNetwork.aocx");
        std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
        cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()+1));
	 programs[0] = new cl::Program(*contexts[0], DeviceList1, mybinaries);
	err = programs[0]->build(DeviceList1);	


	std::ifstream aocx_stream2(f2, std::ios::in|std::ios::binary);
        //checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionNeuralNetwork.aocx");
        std::string prog2(std::istreambuf_iterator<char>(aocx_stream2), (std::istreambuf_iterator<char>()));
        cl::Program::Binaries mybinaries2 (1, std::make_pair(prog2.c_str(), prog2.length()+1));
	programs[1] = new cl::Program(*contexts[1], DeviceList2, mybinaries);
	err = programs[1]->build(DeviceList2);	

	std::string file1_xml = GoogLeNet_DIR+"Inception"+std::to_string(2*rank)+".xml";
	std::string file2_xml = GoogLeNet_DIR+"Inception"+std::to_string(2*rank+1)+".xml";
	
	char f1_xml[file1.length()];
	strcpy(f1,file1_xml.c_str());
	std::vector<std::string> first_kernels;   //kernels from first aocx
	xml_parser1(f1_xml,first_kernels);
	
	char f2_xml[file1.length()];
	strcpy(f2_xml,file2_xml.c_str());
	std::vector<std::string> second_kernels;   //kernels from second aocx
	xml_parser1(f2_xml,second_kernels);
	

	

	//Print the details of each layers in the network to check their correctness.
	//print_layersDetails(cnnLayersList);

	//std::ifstream aocx_stream("/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/designs/inception_modified_nnvm/inception_modified_nnvm.aocx", std::ios::in | std::ios::binary);


	//checkErr(aocx_stream.is_open() ? CL_SUCCESS:-1, overlay_name);
	//std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
	//cl::Program::Binaries mybinaries(1, std::make_pair(prog.c_str(), prog.length() + 1));

	//cl::Program program(mycontext, DeviceList1, mybinaries);

	//err = program.build(DeviceList1);

	//assert(err == CL_SUCCESS);
	//std::cout << " Error code after BUILD:"
			  //<< " is ===>" << err << std::endl;



	cl::Kernel *kernels[250];
	int kernel_index = 0;
	//cl::CommandQueue *queues[50];
	cl::Buffer *buffers[500];
	int buffer_index = 0;




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
    
    for(int i =0;i<3;i++)
    {
        for(int j=0;j<224*224;j++)
        {
            normalized_image[(i*224*224)+j] = float(images[((2-i)*224*224)+j]) * scaling_layer->layerWeights[i] + scaling_layer->layerBias[i];
        }
    }

	float transpose_image[224*224*3];
    for(int i=0;i<224*224;i++)
	{
		for(int j=0;j<3;j++)
		{
			transpose_image[(i*3)+j] = normalized_image[(j*224*224)+i]; 
		}
	}



buffers[buffer_index] = new cl::Buffer(*contexts[0], CL_MEM_READ_ONLY, sizeof(cl_float) * dim_x * dim_y * dim_depth * num_images);
    err = cmd_queues[0]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * dim_x * dim_y * dim_depth * num_images, transpose_image); //images buffer
    assert(err == CL_SUCCESS);
    err = cmd_queues[0]->finish();
	buffer_index++; 
    //std::cout << " Error code after image transfer:" << kernel_index << " is ===>" << err << std::endl;
    assert(err == CL_SUCCESS);

	int num_filters = 0;
	int num_classes = 0;
	int num_pixels = dim_x * dim_y * dim_depth * num_images;
	int padding_kernel_index = 0;
	int padding_out_index = 0;
	//Assing 0 as parent buffer index for root node
	root->parentOutBufferIndex.push_back(0);


	//Flag to launch the layer
	bool launchFlag = true;


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
					const char *layerName = p->layerName.c_str();

					std::cout<<"Launching Layer:"<<layerName<<std::endl;
					int program_number = get_program_num(p->layerName,first_kernels,second_kernels);
					//code to launch kernels
					if (p->layerType == "Convolution"&&program_number!=0)

					{	
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if(ch->visited==1)
							{
								flag_parents++;
							}
						}

						if(!p->visited&&flag_parents==p->parents.size())
						{

						std::cout<<"\t Kernel Index:"<<kernel_index<<std::endl;

						
						//kernels[kernel_index] = new cl::Kernel(*programs[program_number-1], layerName, &err);
						
							
						std::cout << "\t Kernel Created "<<std::endl;
						assert(err == CL_SUCCESS);
						std::cout << "\t  pads_begin :"<<p->params["pads_begin"].at(0)<<","<<p->params["pads_begin"].at(2)<<" pads_end :"<< p->params["pads_begin"].at(0)<<","<<p->params["pads_begin"].at(2) <<std::endl;

						//For zero padding conv layer
						if (p->params["pads_begin"].at(0) == '0' && p->params["pads_begin"].at(2) == '0' && p->params["pads_end"].at(0) == '0' && p->params["pads_end"].at(2) == '0')
						{
							int pad_out_index = 0;
							
							//std::cout<<"\t Kernel Index:"<<kernel_index<<std::endl;
							kernels[kernel_index] = new cl::Kernel(*programs[program_number-1], layerName, &err);
							
							assert(err == CL_SUCCESS);
							
							//output
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
							err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //output of conv
							assert(err == CL_SUCCESS);
							
							p->layerOutBufferIndex = buffer_index;
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							buffer_index++;

							
							err = kernels[kernel_index]->setArg(1, *buffers[p->parentOutBufferIndex.at(0)]); //first argument, input, also the output of the previous layer
							assert(err == CL_SUCCESS);
							buffer_index++;

							//weights
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
							err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
							err = cmd_queues[p->layerID]->finish();
							//std::cout << "\t Error code after weights transfer:" << kernel_index << " is ===>" << err << std::endl;
							assert(err == CL_SUCCESS);
							err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]);
							assert(err == CL_SUCCESS);
							buffer_index++;

							//Bias
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
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
							
							// OUTPUT WRITE BEING //
							if(outputWriteFlag == 1){ 
								float final_labels[p->outH * p->outW * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, final_labels);
								err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								std::ofstream outdataincep;
								std::string outFileName = resultsFileAppender+std::to_string(p->layerID)+"_"+p->layerName+".txt";
								outdataincep.open(outFileName);
								if (!outdataincep)
								{ // file couldn't be opened
									std::cerr << "Error: file could not be opened" << std::endl;
										exit(1);
								}
								//std::cout << "\tConcat output\n";
								for (int i = 0; i < p->outH * p->outW * p->outDepth; i++)
								{
									//std::cout << final_labels[i] << " ";
									outdataincep << final_labels[i] << "\n";
								}
								outdataincep.close(); 
							}
							// OUTPUT WRITE END //
							
							assert(err == CL_SUCCESS);
							p->visited = 1;
						}
						else
						{
							int inputIndex= 0;
							if(root->parents.size() == 0){
								inputIndex = 0;
							}else{
								inputIndex = p->parentOutBufferIndex.at(0);
							}
							//Pad kernel launching code

							std::string pad_kernel_name = "Padding_"+p->layerName;
							const char *pad_name = pad_kernel_name.c_str();
							//std::cout<<"\t Kernel Index:"<<kernel_index<<std::endl;
							kernels[kernel_index] = new cl::Kernel(*programs[program_number-1], pad_name, &err);
							
							assert(err == CL_SUCCESS);

							//buffer_index++;
							//std::cout << "\tbuffer index:"<<p->parentOutBufferIndex.at(0)<<std::endl;

							//weights
							//buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
							//err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
							//err = cmd_queues[p->layerID]->finish();
							//std::cout << "\t Error code after weights transfer:" << kernel_index << " is ===>" << err << std::endl;

							int pad_x = p->params["pads_begin"].at(0) - '0';
							int pad_y = p->params["pads_end"].at(0) - '0';
							
							int dim1 = p->outH+pad_x+pad_y;
							int dim2 = p->outW+pad_x+pad_y;
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_WRITE, sizeof(cl_float) * dim1 * dim2 * p->outDepth);
							int pad_out_index = buffer_index;
							err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]);//err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //output of pad							

							assert(err == CL_SUCCESS);
							buffer_index++;							
							err = kernels[kernel_index]->setArg(1, *buffers[p->parentOutBufferIndex.at(0)]);	//input to pad 					
							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							assert(err == CL_SUCCESS);

							//buffer_index++;
							//std::cout << "\tweights passed\n";

							//Bias
							//buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
							//err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases

							err = cmd_queues[p->layerID]->finish();
							
							// OUTPUT WRITE BEING //
							if(outputWriteFlag == 1){ 
								float final_labels[dim1 * dim2 * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[pad_out_index], CL_TRUE, 0, sizeof(cl_float) * dim1 * dim2 * p->outDepth, final_labels);
								err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								std::ofstream outdataincep;
								std::string outFileName = resultsFileAppender+std::to_string(p->layerID)+"_"+pad_kernel_name+".txt";
								outdataincep.open(outFileName);
								if (!outdataincep)
								{ // file couldn't be opened
									std::cerr << "Error: file could not be opened" << std::endl;
										exit(1);
								}
								std::cout << "\tConcat output\n";
								for (int i = 0; i < p->outH * p->outW * p->outDepth; i++)
								{
									//std::cout << final_labels[i] << " ";
									outdataincep << final_labels[i] << "\n";
								}
								outdataincep.close(); 
							}
							// OUTPUT WRITE END //
							
							assert(err == CL_SUCCESS);
							kernel_index++;



							// Pad kernel launching code ends
							//std::cout<<"\t Kernel Index:"<<kernel_index<<std::endl;
							kernels[kernel_index] = new cl::Kernel(*programs[program_number-1], layerName, &err);
							//std::cout << "\t Kernel Created "<<std::endl;
							assert(err == CL_SUCCESS);
							
							//output

							buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
							err = kernels[kernel_index]->setArg(4, *buffers[buffer_index]); //output of conv

							assert(err == CL_SUCCESS);
							std::cout << "\tconv output passed :"
									  << p->outH * p->outW * p->outDepth << std::endl;
							p->layerOutBufferIndex = buffer_index;
							if(p->children.size()>0)
							{
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							}
							buffer_index++;




							//input
							err = kernels[kernel_index]->setArg(1, *buffers[pad_out_index]); //first argument, input, also the output of the previous layer
							//err = kernels[kernel_index]->setArg(0, *buffers[p->parents.at(0)->layerOutBufferIndex]);
							assert(err == CL_SUCCESS);
							buffer_index++;
							//std::cout << "\tbuffer index:"<<p->parentOutBufferIndex.at(0)<<std::endl;

							//weights
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
							err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
							err = cmd_queues[p->layerID]->finish();
							//std::cout << "\t Error code after weights transfer:" << kernel_index << " is ===>" << err << std::endl;
							assert(err == CL_SUCCESS);
							err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]);
							assert(err == CL_SUCCESS);
							buffer_index++;
							//std::cout << "\tweights passed\n";

							//Bias
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
							err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
							err = cmd_queues[p->layerID]->finish();
							//std::cout << "\t Error code after bias transfer:" << kernel_index << " is ===>" << err << std::endl;
							assert(err == CL_SUCCESS);
							err = kernels[kernel_index]->setArg(3, *buffers[buffer_index]);
							assert(err == CL_SUCCESS);
							buffer_index++;
							//std::cout << "\tbiases passed\n";
							//Num of images
							//err = kernels[kernel_index]->setArg(3, 1);
							
							//assert(err == CL_SUCCESS); 
							//std::cout << "\tconv out dim :"
							//		  << p->outH * p->outW * p->outDepth << std::endl;
							
							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
							assert(err == CL_SUCCESS);
							//std::cout << "\t Error code soon after conv layer for kernel:" << kernel_index << " is ===>" << err << std::endl;
							err = cmd_queues[p->layerID]->finish();
							assert(err == CL_SUCCESS);
							
							// OUTPUT WRITE BEING //
							if(outputWriteFlag == 1){ 
								float final_labels[p->outH * p->outW * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, final_labels);
								err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								std::ofstream outdataincep;
								std::string outFileName = resultsFileAppender+std::to_string(p->layerID)+"_"+p->layerName+".txt";
								outdataincep.open(outFileName);
								if (!outdataincep)
								{ // file couldn't be opened
									std::cerr << "Error: file could not be opened" << std::endl;
										exit(1);
								}
								std::cout << "\tConcat output\n";
								for (int i = 0; i < p->outH * p->outW * p->outDepth; i++)
								{
									//std::cout << final_labels[i] << " ";
									outdataincep << final_labels[i] << "\n";
								}
								outdataincep.close(); 
							}
							// OUTPUT WRITE END //
							
							kernel_index++;
							p->visited = 1;
						}
						//std::cout << "\t Error code  after conv layer finish for kernel:" << kernel_index << " is ===>" << err << std::endl;
						//std::cout << "\t Input buffer index:" << p->parentOutBufferIndex.at(0)  << std::endl;
						//std::cout << "\t Output buffer index:" << p->layerOutBufferIndex  << std::endl;
						
						//Stop the execution after last conv
						if(p->layerName == "Conv2d_0c_1x1_Conv2D"){
							//Read buffer from the last conv output
							float convScores[p->outH * p->outW * p->outDepth];
							cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, convScores);
							err = cmd_queues[p->layerID]->finish();

							//Printing the results after getting Top N Results
							std::cout<<" TOP- "<<TOP_N<< " Classification"<<std::endl;
							std::cout<<" --------------------------------"<<std::endl;
							std::vector<int> results= getTopNResults(convScores,TOP_N);
							std::cout<<" --------------------------------"<<std::endl;
							std::cout<<" Please match the above labels with the \"Labels.txt\" of the model to see the classification results."<<std::endl;
							return 0;
						}

						}
							if(p->visited==0)
								q.push(p);
					}

					else if (p->layerType == "Pooling"&&program_number!=0)

					{
						// To check on which device this  layer is mapped to.
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if(ch->visited==1)
							{
								flag_parents++;
							}
						}
						if(!p->visited&&flag_parents==p->parents.size())
						{

						
						std::cout << "\t kernel:"<<layerName<<std::endl;
						
						kernels[kernel_index] = new cl::Kernel(*programs[program_number-1], layerName, &err);
						
							
						assert(err == CL_SUCCESS);
						//output
						buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);
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
						//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
						assert(err == CL_SUCCESS);
						cmd_queues[p->layerID]->finish();
						assert(err == CL_SUCCESS);
						
						// OUTPUT WRITE BEING //
							if(outputWriteFlag == 1){ 
								float final_labels[p->outH * p->outW * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, final_labels);
								err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								std::ofstream outdataincep;
								std::string outFileName = resultsFileAppender+std::to_string(p->layerID)+"_"+p->layerName+".txt";
								outdataincep.open(outFileName);
								if (!outdataincep)
								{ // file couldn't be opened
									std::cerr << "Error: file could not be opened" << std::endl;
										exit(1);
								}
								std::cout << "\tConcat output\n";
								for (int i = 0; i < p->outH * p->outW * p->outDepth; i++)
								{
									//std::cout << final_labels[i] << " ";
									outdataincep << final_labels[i] << "\n";
								}
								outdataincep.close(); 
							}
							// OUTPUT WRITE END //
						
						kernel_index++;
						p->visited = 1;
						//std::cout << "\t Input buffer index:" << p->parentOutBufferIndex.at(0)  << std::endl;
						//std::cout << "\t Output buffer index:" << p->layerOutBufferIndex  << std::endl;
						
						}
						if(p->visited==0)
							q.push(p);
					}

					else if (p->layerType == "FullyConnected"&&program_number!=0)

					{
						// To check on which device this  layer is mapped to.
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if(ch->visited==1)
							{
								flag_parents++;
							}
						}
						if(!p->visited&&flag_parents==p->parents.size())
						{
						
							kernels[kernel_index] = new cl::Kernel(*programs[program_number-1], layerName, &err);
						
						assert(err == CL_SUCCESS);

						buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_weights);
						err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_weights, p->layerWeights); //weights
						cmd_queues[p->layerID]->finish();
						assert(err == CL_SUCCESS);
						err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]);
						assert(err == CL_SUCCESS);
						buffer_index++;
						std::cout << "\tweights passed\n";
						buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_ONLY, sizeof(cl_float) * p->num_biases);
						err = cmd_queues[p->layerID]->enqueueWriteBuffer(*buffers[buffer_index], CL_FALSE, 0, sizeof(cl_float) * p->num_biases, p->layerBias); //biases
						cmd_queues[p->layerID]->finish();
						assert(err == CL_SUCCESS);
						err = kernels[kernel_index]->setArg(1, *buffers[buffer_index]);
						assert(err == CL_SUCCESS);
						buffer_index++;
						std::cout << "\tbiases passed\n";

						buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_WRITE, sizeof(cl_int) * num_images);
						err = kernels[kernel_index]->setArg(2, *buffers[buffer_index]); //output of FC
						assert(err == CL_SUCCESS);
						std::cout << "\toutput of FC passed\n";
						p->layerOutBufferIndex = buffer_index;
						for (struct layersDetails *ch : p->children)
						{
							ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
						}
						buffer_index++;
						err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
							assert(err == CL_SUCCESS);
							cmd_queues[p->layerID]->finish();
							kernel_index++;
						p->visited = 1;
						}
						if(p->visited==0)
							q.push(p);
					}

					else if (p->layerType == "Concat"&&program_number!=0)

					{
						//std::cout << " \tLaunching concat\n";
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if(ch->visited==1)
							{
								flag_parents++;
							}
						}
						
						if(!p->visited&&flag_parents==p->parents.size())
						{
							//std::cout << "\t concat not visited earlier\n";
							//std::cout << "\t Input buffer index:" << p->parents.at(0)->layerOutBufferIndex  <<", "<<p->parents.at(1)->layerOutBufferIndex <<", "<<p->parents.at(2)->layerOutBufferIndex  <<", "<<p->parents.at(4)->layerOutBufferIndex  <<", " << std::endl;
							
							kernels[kernel_index] = new cl::Kernel(*programs[program_number-1], layerName, &err);
							
							assert(err == CL_SUCCESS);

							std::cout<<"\tConcat Parents:"<<p->parents.size()<<std::endl;
							std::cout << "\toutput\n";
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);

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
							//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
							assert(err == CL_SUCCESS);
							cmd_queues[p->layerID]->finish();
							//std::cout<<"Input Buffer Index : ##"<<  p->parents.at(0)->layerOutBufferIndex <<"  " << p->parents.at(1)->layerOutBufferIndex<<std::endl ;
							//std::cout<<"Input Buffer Index : ##"<<  p->parents.at(2)->layerOutBufferIndex <<"  " << p->parents.at(3)->layerOutBufferIndex<<std::endl ;
							//std::cout << "\t Output buffer index:" << p->layerOutBufferIndex  << std::endl;
							kernel_index++;
							p->visited = 1;
							
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
											
							// OUTPUT WRITE BEING //
							if(outputWriteFlag == 1){ 
								float final_labels[p->outH * p->outW * p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outW * p->outDepth, final_labels);
								err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								std::ofstream outdataincep;
								std::string outFileName = resultsFileAppender+std::to_string(p->layerID)+"_"+p->layerName+".txt";
								outdataincep.open(outFileName);
								if (!outdataincep)
								{ // file couldn't be opened
									std::cerr << "Error: file could not be opened" << std::endl;
										exit(1);
								}
								//std::cout << "\tConcat output\n";
								for (int i = 0; i < p->outH * p->outW * p->outDepth; i++)
								{
									//std::cout << final_labels[i] << " ";
									outdataincep << final_labels[i] << "\n";
								}
								outdataincep.close(); 
							}
							// OUTPUT WRITE END //
							
												
						}
						if(p->visited == 0)
							q.push(p);
						
						

					}
					else if(p->layerType == "SoftMax"&&program_number!=0)
					{	
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if(ch->visited==1)
							{
								flag_parents++;
							}
						}
						if(!p->visited&&flag_parents==p->parents.size())
						{
						std::cout << "\t Input buffer index:" << p->parentOutBufferIndex.at(0)  << std::endl;
						std::cout << " outH, OutW, outDepth:"<<p->outH << ","<<p->outW<<","<<p->outDepth<<std::endl;
						
							kernels[kernel_index] = new cl::Kernel(*programs[program_number-1], layerName, &err);
						
						assert(err == CL_SUCCESS);

						buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outW * p->outDepth);

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
						//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
						assert(err == CL_SUCCESS);

						kernel_index++;
						p->visited = 1;	

						cmd_queues[p->layerID]->finish();
						kernel_index++;	
						
						std::cout << "\t Output buffer index:" << p->layerOutBufferIndex  << std::endl;

						// OUTPUT WRITE BEING //
							if(outputWriteFlag == 1){ 
								float final_labels[p->outDepth];
								cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex], CL_TRUE, 0, sizeof(cl_float) * p->outDepth, final_labels);
								err = cmd_queues[p->layerID]->finish();
								assert(err == CL_SUCCESS);
								std::ofstream outdataincep;
								std::string outFileName = resultsFileAppender+std::to_string(p->layerID)+"_"+p->layerName+".txt";
								outdataincep.open(outFileName);
								if (!outdataincep)
								{ // file couldn't be opened
									std::cerr << "Error: file could not be opened" << std::endl;
										exit(1);
								}
								std::cout << "\tConcat output\n";
								for (int i = 0; i < p->outDepth; i++)
								{
									//std::cout << final_labels[i] << " ";
									outdataincep << final_labels[i] << "\n";
								}
								outdataincep.close(); 
							}
							// OUTPUT WRITE END //
							//Stop execution after softmax
							if(p->layerName == "Predictions_Softmax")
								exit(0);

						}
						if(p->visited==0)
							q.push(p);
					}
					else if (p->layerType == "Reshape"&&program_number!=0)
					{
						int flag_parents = 0;
						for (struct layersDetails *ch : p->parents)
						{
							if(ch->visited==1)
							{
								flag_parents++;
							}
						}						
						if(!p->visited&&flag_parents==p->parents.size())
						{
						
							kernels[kernel_index] = new cl::Kernel(*programs[program_number-1], layerName, &err);
						
						assert(err == CL_SUCCESS);
						if (p->layerName == "Predictions_Reshape")
						{
							std::cout << "\t Input buffer index:" << p->parentOutBufferIndex.at(0)  << std::endl;
							std::cout << " outH, OutW, outDepth:"<<p->outH << ","<<p->outW<<","<<p->outDepth<<std::endl;
							buffers[buffer_index] = new cl::Buffer(*contexts[program_number-1], CL_MEM_READ_WRITE, sizeof(cl_float) * p->outH * p->outDepth);
							err = kernels[kernel_index]->setArg(0, *buffers[buffer_index]); //reshape output
							assert(err == CL_SUCCESS);
							p->layerOutBufferIndex = buffer_index;
							if(p->children.size()>0)
							{
							for (struct layersDetails *ch : p->children)
							{
								ch->parentOutBufferIndex.push_back(p->layerOutBufferIndex);
							}
							}
							buffer_index++;
							std::cout<<"No of parents: "<<p->parents.size()<<"\n";
							//input
							err = kernels[kernel_index]->setArg(1, *buffers[p->parents.at(0)->layerOutBufferIndex]); //conv input1
							assert(err == CL_SUCCESS);
							buffer_index++;
							err = kernels[kernel_index]->setArg(2, *buffers[p->parents.at(1)->layerOutBufferIndex]); //conv input2
							assert(err == CL_SUCCESS);
							buffer_index++;
							err = cmd_queues[p->layerID]->enqueueTask(*kernels[kernel_index]);
							//std::this_thread::sleep_for(std::chrono::milliseconds(5000));
							assert(err == CL_SUCCESS);

							kernel_index++;
							p->visited = 1;

							cmd_queues[p->layerID]->finish();
							
							std::cout << "\t Output buffer index:" << p->layerOutBufferIndex  << std::endl;

							float final_labels[ p->outH * p->outDepth];
							cmd_queues[p->layerID]->enqueueReadBuffer(*buffers[p->layerOutBufferIndex ], CL_TRUE, 0, sizeof(cl_float) * p->outH * p->outDepth, final_labels);
							
							std::cout<<"\tLabels top 10\n";
							for(int i=0;i<10;i++)
								std::cout<<final_labels[i]<<"\n";
					
						}
						else
						{
							std::cout << "\t No supporting Reshape layer" << std::endl;
						}
						}
						if(p->visited==0)
							q.push(p);
					}
				}
				// Enqueue all children of the dequeued item
				if(p->visited == 1)
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

	return 0;
}




std::vector<int> getTopNResults(float final_labels[],int topN){
	std::vector<int> results;
	std::multimap<float, int , std::greater<float>> sorted_map;
	int N = 0;
	for(int i=0;i<1001;i++){
		//convScores.insert(i,final_labels[i]);
		sorted_map.insert(std::make_pair(final_labels[i],i));
	}

 	


	for (auto entry: sorted_map)
    {
		if(N<topN){
			results.push_back(entry.second);
        	std::cout << "Label Number: "<<entry.second << " - Score " << entry.first <<std::endl;
		}else{
			break;
		}
		N++;
    }
	return results;

}

