#include "CL/cl.hpp"
#include <inference_engine.hpp>
#include <string>
#include <iostream>

using namespace InferenceEngine;



/**
 * function to check if the layer is supported by the plugin.
 * input : Layer Name
 * output : True/False 
 */ 

bool isLayerSupported(std::string layer_name);
/**
 * function to check if the layer is already added in the Tree.
 * input : Layer ID
 * output : True/False
 */
bool isDuplicate(int id);

/**
 *  Rename the layer name to match it with kernels. 
 *  Here we replace '/' from layer name in the IR to '_'
 *  Remove "InceptionV1/InceptionV1/" from the name
 * input : Layer Name, delimiter 
 * output : new layer name 
 */
std::string rename_node_name(std::string strToSplit, char delimiter);
/**
 * Function to print Images
 * input : pointer to the image, number of images, height,width
 */
void printImage(unsigned char *image, int numberOfImages, int xdim, int ydim);

/**
 * Parse the input images using OpenCV
 * input: filepath to image,cnn network
 */
void parse_images(std::vector<std::string> imageNames, InferenceEngine::CNNNetwork network);

/**
 * function to check if the input model's bitstream is present in the Noctua FPGA Bitstream Repo.
 */
std::string bitstreamFinder(char *filepath);
/**
 * Tree Construction logic:
 */
struct layersDetails *parse_root(InferenceEngine::CNNNetwork network);
/**
 * Find a layer by its ID
 */
void findbyID(struct layersDetails *root, int id, struct layersDetails *parent);


/**
 * fucntion to remove dummy layers from the tree.
 */
void remove_dummy_child(struct layersDetails *node);

/**
 * Level order traversal to find node with particular ID
 */
void find_missing_duplicates(struct layersDetails *root);

/**
 * TREE construction logic:
 */
struct layersDetails *parse_child(InferenceEngine::CNNNetwork network, std::string layer_name, struct layersDetails *root,struct layersDetails *parent);

/**
 * Print the constructed tree
 */
void printCNNTree(layersDetails *root);

/**
 * Print all the available Platforms
 */
void printPlatforms(std::vector<cl::Platform> PlatformList);

/**
 * Print all the devices.
 */
void printDevices(std::vector<cl::Device> DeviceList1);

/**
 * OPENVINO FPGA NOCTUA PLUGIN is implemented in this function  
 */
std::vector<int> fpga_launcher(InferenceEngine::CNNNetwork network, char *model_path, std::vector<std::string> imageNames, std::string model_name, int rank, int TOP_N, std::string bitstream_dir, std::string opencl_design, std::string route_xml_path);
/**
 *  Launcher function for OpenCL Global mem design
 */
std::vector<int> launcher_global(std::vector<cl::Device> DeviceList1,
		std::vector<cl::Device> DeviceList2, int kernel_index, cl_int err,
		int buffer_index, int rank, int com_sz, struct layersDetails* root,
		cl::CommandQueue* cmd_queues[250], cl::Context* contexts[2],
		cl::Kernel* kernels[250], cl::Program* programs[2],
		cl::Buffer* buffers[500] , std::vector<std::string> first_kernels,std::vector<std::string> second_kernels, int TOP_N, std::string model_name);

/**
 *  Launcher function for OpenCL Channels design
 */
std::vector<int> launcher_channel(int kernel_index,
								  int buffer_index, int TOP_N, std::map<std::string, unsigned int> route_map,
								  std::vector<cl::Device> DeviceList1, std::vector<cl::Device> DeviceList2,
								  struct layersDetails *root, cl::Kernel *kernels[250],
								  cl::Program *programs[2], cl::Buffer *buffers[500],
								  cl::Context *contexts[2], cl::CommandQueue *cmd_queues[500],
								  cl::CommandQueue *pad_queues[500], int rank, std::vector<std::string> first_kernels, std::vector<std::string> second_kernels);


/**
 * Function to get Top 10 labels
 */
std::vector<int> getTopNResults(float final_labels[],int topN);

/**
 * Topology of FPGAs
 * To be built using xml parser. This is just a dummy placeholder
 */
std::map<std::string, unsigned int> build_topo_map(std::string xmlpath);