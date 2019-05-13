#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include "CL/cl.hpp"
#include "utility.h"
#include <fstream>
#include <chrono>

static const int NUMBER_OF_IMAGES  = 3;
static const int NUMBER_OF_PIXELS = 100;
static const int NUMBER_OF_CLASSES = 5;  //0 to 4
static const int NUMBER_OF_FILTERS = 4;
static const int NUMBER_OF_ROWS = 10; //Including zero padding
static const int NUMBER_OF_COLS = 10; //Including zero padding
static const int FILTER_ROWS = 3; // Number of rows in the conv Filter
static const int FILTER_COLS = 3; // Number of rows in the conv Filter
static const int CONV_PADDING = 1; // Number of Zero Padding
static const int CONV_STRIDE=1; // Stride
static const int CONV_OUTPUT_ROWS = 10; // Number of Rows in the output image from Maxpool
static const int CONV_OUTPUT_COLS = 10;  // Number of Cols in the output image from Maxpool
static const int MAXPOOL_OUTPUT_ROWS = 5; // Number of Rows in the output image from Maxpool
static const int MAXPOOL_OUTPUT_COLS = 5;  // Number of Cols in the output image from Maxpool
static const int CONCAT_OUTPUT_ROWS = 5; // Number of Rows in the output image from Concat
static const int CONCAT_OUTPUT_COLS = 5;  // Number of Cols in the output image from Concat
static const int CONCAT_NUMBER_OF_FILTERS = 8;
unsigned char calculatedLabels[NUMBER_OF_IMAGES];// Classified Class after FC
int kernelcalculatedLabels[NUMBER_OF_IMAGES];
unsigned char available_labels[NUMBER_OF_IMAGES]; // Labels from the MNIST Dataset
static const int TOTAL_NUMBER_OF_IMAGE_PIXELS = NUMBER_OF_IMAGES*NUMBER_OF_ROWS*NUMBER_OF_COLS;
static const int TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS = NUMBER_OF_FILTERS*FILTER_ROWS*FILTER_COLS;
static const int TOTAL_NUMBER_OF_CONV_OUT_PIXELS=NUMBER_OF_IMAGES*NUMBER_OF_FILTERS*CONV_OUTPUT_ROWS*CONV_OUTPUT_COLS;
static const int NUMBER_OF_FC_PIXELS =NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS;
static const int NUMBER_OF_FC_WEIGHTS =NUMBER_OF_FC_PIXELS*NUMBER_OF_CLASSES;
static const int NUMBER_OF_PIXELS_FCL = 175 ;

char Kernel_Img[TOTAL_NUMBER_OF_IMAGE_PIXELS] __attribute__ ((aligned (64)));
short Kernel_CNN_WEIGHTS[TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS]  __attribute__ ((aligned (64)));
short Kernel_CNN_BIAS[NUMBER_OF_FILTERS]  __attribute__ ((aligned (64)));
int Kernel_Out[TOTAL_NUMBER_OF_CONV_OUT_PIXELS]  __attribute__ ((aligned (64)));

int Conv_Output_local[NUMBER_OF_IMAGES*32*NUMBER_OF_PIXELS] __attribute__ ((aligned (64)));
int Maxpool_Output[NUMBER_OF_IMAGES*NUMBER_OF_PIXELS_FCL] __attribute__ ((aligned (64)));





int main(void)
{
std::cout << "started"<< std::endl;
        // FPGA Implementation
        cl_int err;

        //Setup Platform

        //Get Platform ID
        std::vector<cl::Platform> PlatformList;
        ////////////// Exercise 1 Step 2.3
        err = cl::Platform::get(&PlatformList);
        assert(err==CL_SUCCESS);
        checkErr(PlatformList.size()==1 ? CL_SUCCESS : -1, "cl::Platform::get");
        print_platform_info(&PlatformList);

        //Setup Device
        //Get Device ID
        std::vector<cl::Device> DeviceList;
        err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList);
	std::cout << err << std::endl;        
	assert(err==CL_SUCCESS);
        print_device_info(&DeviceList);

        //Create Context
        cl::Context mycontext(DeviceList);
        assert(err==CL_SUCCESS);

        //Create Command queue
        cl::CommandQueue queueConvLayer(mycontext, DeviceList[0]);
        assert(err==CL_SUCCESS);
        cl::CommandQueue queueMaxPool(mycontext, DeviceList[0]);
        assert(err==CL_SUCCESS);
        cl::CommandQueue queueAvgPool(mycontext, DeviceList[0]);
        assert(err==CL_SUCCESS);
        cl::CommandQueue queueConcat(mycontext, DeviceList[0]);
        assert(err==CL_SUCCESS);
        cl::CommandQueue queueFCLayer(mycontext, DeviceList[0]);
        assert(err==CL_SUCCESS);
        cl::CommandQueue queueSoftMax(mycontext, DeviceList[0]);
        assert(err==CL_SUCCESS);

	//Create Buffers for input and outputs
        cl::Buffer Buffer_Imgs(mycontext, CL_MEM_READ_ONLY, sizeof(char)* TOTAL_NUMBER_OF_IMAGE_PIXELS);
        cl::Buffer Buffer_ConvWeights(mycontext, CL_MEM_READ_ONLY, sizeof(short)* TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS);
        cl::Buffer Buffer_ConvBias(mycontext, CL_MEM_READ_ONLY, sizeof(short)* NUMBER_OF_FILTERS);
	cl::Buffer Buffer_ConvOutput(mycontext,CL_MEM_READ_WRITE,sizeof(int)* NUMBER_OF_IMAGES*CONV_OUTPUT_ROWS*CONV_OUTPUT_COLS*NUMBER_OF_FILTERS);

	cl::Buffer Buffer_MaxPoolOutput(mycontext, CL_MEM_READ_WRITE, sizeof(int)* NUMBER_OF_PIXELS_FCL * NUMBER_OF_IMAGES);
	cl::Buffer Buffer_AvgPoolOutput(mycontext, CL_MEM_READ_WRITE, sizeof(int)* NUMBER_OF_PIXELS_FCL * NUMBER_OF_IMAGES);
	cl::Buffer Buffer_ConcatInput(mycontext, CL_MEM_READ_ONLY, sizeof(int)* NUMBER_OF_PIXELS_FCL * NUMBER_OF_IMAGES * 2);
	cl::Buffer Buffer_ConcatLayers(mycontext, CL_MEM_READ_ONLY, sizeof(int)* 2);
	cl::Buffer Buffer_ConcatOutput(mycontext, CL_MEM_READ_WRITE, sizeof(int)* NUMBER_OF_PIXELS_FCL * NUMBER_OF_IMAGES * 2);
	
	cl::Buffer Buffer_FCLWeights(mycontext, CL_MEM_READ_ONLY, sizeof(short)*NUMBER_OF_FC_WEIGHTS);
	cl::Buffer Buffer_FCLBias(mycontext, CL_MEM_READ_ONLY, sizeof(short)*NUMBER_OF_CLASSES);
        cl::Buffer Buffer_FCLOutput(mycontext,CL_MEM_READ_WRITE,sizeof(int)*NUMBER_OF_IMAGES);
	cl::Buffer Buffer_SoftMaxOutput(mycontext,CL_MEM_READ_ONLY,sizeof(int)*NUMBER_OF_IMAGES);

	
	
	std::cout << "Creating the Images" << std::endl;
	for (int i; i < NUMBER_OF_PIXELS*NUMBER_OF_IMAGES; i++) {
		Kernel_Img[i] = i%3;	
	}

	std::cout << "Finished creating the Images" << std::endl;

	std::cout << "Creating the Class Weights" << std::endl;
        short digitWeights[NUMBER_OF_CLASSES*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS*NUMBER_OF_FILTERS];
	int temp_count = 0;
        for(int i=0; i<NUMBER_OF_CLASSES; i++) {
                for(int j=0; j<(CONCAT_OUTPUT_ROWS*CONCAT_OUTPUT_COLS*CONCAT_NUMBER_OF_FILTERS); j++) {
			digitWeights[temp_count] = i;

                }
        }
        std::cout << "Finished creating the Class Weights" << std::endl;

	std::cout << "Creating the Convolution Weights" << std::endl;
	temp_count = 0;
        for(int i=0; i<NUMBER_OF_FILTERS; i++) {
                for(int j=0; j<FILTER_ROWS; j++) {
                        for(int k=0; k<FILTER_COLS; k++) {
                                Kernel_CNN_WEIGHTS[temp_count]= i+1;
				temp_count++;
                        }
                }
        }
        //Bias Array
        for(int i=0; i<NUMBER_OF_FILTERS; i++)
                Kernel_CNN_BIAS[i]= i;
	std::cout << "Finished creating the Convolution Weights" << std::endl;
/*
	//Write data to device
        err = queueConvLayer.enqueueWriteBuffer(Buffer_Imgs, CL_FALSE, 0, sizeof(char)*TOTAL_NUMBER_OF_IMAGE_PIXELS, Kernel_Img);
        assert(err==CL_SUCCESS);
        err = queueConvLayer.enqueueWriteBuffer(Buffer_ConvWeights, CL_FALSE, 0, sizeof(short)*TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS, Kernel_CNN_WEIGHTS);
        assert(err==CL_SUCCESS);
        err = queueConvLayer.enqueueWriteBuffer(Buffer_ConvBias, CL_FALSE, 0, sizeof(short)*NUMBER_OF_FILTERS, Kernel_CNN_BIAS);
        assert(err==CL_SUCCESS);
        err = queueFCLayer.enqueueWriteBuffer(Buffer_FCLWeights, CL_FALSE, 0, sizeof(short)*NUMBER_OF_FC_WEIGHTS, digitWeights);
        assert(err==CL_SUCCESS);

	// create the kernel
        const char *CONV_kernel_name = "ConvolutionLayer";
        const char *MP_kernel2_name = "MaxPool";
        const char *FC_kernel3_name = "FCL_Kernel";
        //Read in binaries from file
        std::ifstream aocx_stream("./SimpleCNN.aocx", std::ios::in|std::ios::binary);
        checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "SimpleCNN.aocx");
        std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
        cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()+1));	

        // Create the Program from the AOCX file.
        cl::Program program(mycontext, DeviceList, mybinaries);

        // build the program
        err=program.build(DeviceList);
        assert(err==CL_SUCCESS);
        // create the kernel
        cl::Kernel kernel(program, CONV_kernel_name, &err);
        assert(err==CL_SUCCESS);

        cl::Kernel kernel2(program,MP_kernel2_name, &err);
        assert(err==CL_SUCCESS);

        cl::Kernel kernel3(program,FC_kernel3_name, &err);
        assert(err==CL_SUCCESS);

        //////////////     Set Arguments to the Kernels
        err = kernel.setArg(0, Buffer_Imgs);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(1, Buffer_ConvWeights);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(2, Buffer_ConvBias);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(3, FILTER_ROWS);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(4, FILTER_COLS);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(5, NUMBER_OF_FILTERS);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(6, NUMBER_OF_IMAGES);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(7, CONV_LAYER_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(8, CONV_LAYER_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(9, ZERO_PADDING);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(10, STRIDE);
        assert(err==CL_SUCCESS);
	err = kernel.setArg(11, Buffer_ConvOutput);
        assert(err==CL_SUCCESS);

	 // Launch Kernel
        err=queueConvLayer.enqueueTask(kernel);
        assert(err==CL_SUCCESS);


	//err=queueConvLayer.enqueueReadBuffer(Conv_output, CL_TRUE, 0, sizeof(int)*(NUMBER_OF_PIXELS * NUMBER_OF_IMAGES * 32), Conv_Output_local);
        //assert(err==CL_SUCCESS);

	err=queueConvLayer.finish();
        assert(err==CL_SUCCESS);

	printf("\nLaunching the kernel1 conv...\n");

	//err = queueMaxPool.enqueueWriteBuffer(ConvMaxPoolBuffer, CL_FALSE, 0, sizeof(int)*(NUMBER_OF_PIXELS * NUMBER_OF_IMAGES * 32), Conv_Output_local);
        //assert(err==CL_SUCCESS);

	err = kernel2.setArg(0, Buffer_ConvOutput);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(1, CONV_LAYER_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(2, CONV_LAYER_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(3, NUMBER_OF_FILTERS);
        assert(err==CL_SUCCESS);
	err = kernel2.setArg(4,STRIDE);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(5, NUMBER_OF_IMAGES);
        assert(err==CL_SUCCESS);
	err = kernel2.setArg(6, Buffer_MaxPoolOutput);
        assert(err==CL_SUCCESS);

	err=queueMaxPool.enqueueTask(kernel2);
        assert(err==CL_SUCCESS);

	//err=queueMaxPool.enqueueReadBuffer(maxOutput, CL_TRUE, 0, sizeof(int)*(NUMBER_OF_PIXELS_FCL * NUMBER_OF_IMAGES), Maxpool_Output);
        //assert(err==CL_SUCCESS);

	err=queueMaxPool.finish();
        assert(err==CL_SUCCESS);

        //err = queueFCLayer.enqueueWriteBuffer(MaxFCLBuffer, CL_FALSE, 0, sizeof(int)*(NUMBER_OF_PIXELS_FCL * NUMBER_OF_IMAGES), Maxpool_Output);
        //assert(err==CL_SUCCESS);

	err = kernel3.setArg(0, Buffer_MaxPoolOutput);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(1, Buffer_FCLWeights);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(2, NUMBER_OF_FC_PIXELS);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(3, NUMBER_OF_CLASSES);
        assert(err==CL_SUCCESS);
	err = kernel3.setArg(4, NUMBER_OF_IMAGES);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(5, Buffer_FCLOutput);
        assert(err==CL_SUCCESS);
	err = kernel3.setArg(6, MAXPOOL_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
	err = kernel3.setArg(7, MAXPOOL_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
	err = kernel3.setArg(8, NUMBER_OF_FILTERS);
        assert(err==CL_SUCCESS);

        printf("\nLaunching the kernel...\n");

	auto startFPGA = std::chrono::high_resolution_clock::now();

        err=queueFCLayer.enqueueTask(kernel3);
        assert(err==CL_SUCCESS);

        // read the output
        err=queueFCLayer.enqueueReadBuffer(Buffer_FCLOutput, CL_TRUE, 0, sizeof(int)*NUMBER_OF_IMAGES, kernelcalculatedLabels);
        assert(err==CL_SUCCESS);
        
        err=queueFCLayer.finish();
        assert(err==CL_SUCCESS);

	auto endFPGA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedFPGA = endFPGA - startFPGA;
        std::cout << "FPGA ==> Time Taken for Convolution of 10k images and 32 filters (in sec) :" <<elapsedFPGA.count()<< std::endl;

	float counterfpga = 0;
        for(int zc = 0; zc < NUMBER_OF_IMAGES; zc++)
        {
                if(kernelcalculatedLabels[zc] == (int)available_labels[zc])
                        counterfpga++;

                //if(zc<100)
                  //      std::cout << "FPGA Value :" <<kernelcalculatedLabels[zc] << " ,Label"<<(int)available_labels[zc] << '\n';
        }

        std::cout << "Number of Images correctly classified: " << counterfpga <<std::endl;
        float p_f_Accuracy = (counterfpga/ NUMBER_OF_IMAGES) * 100;

        printf("FPGA Accuracy is %f\n",p_f_Accuracy);
*/
}
