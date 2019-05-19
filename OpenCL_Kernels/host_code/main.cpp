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
static const int DEPTH = 1;
static const int NUMBER_OF_ROWS = 9; //Including zero padding
static const int NUMBER_OF_COLS = 9; //Including zero padding
static const int FILTER_ROWS = 3; // Number of rows in the conv Filter
static const int FILTER_COLS = 3; // Number of rows in the conv Filter
static const int CONV_PADDING = 1; // Number of Zero Padding
static const int CONV_STRIDE=1; // Stride
static const int CONV_OUTPUT_ROWS = 9; // Number of Rows in the output image from Maxpool
static const int CONV_OUTPUT_COLS = 9;  // Number of Cols in the output image from Maxpool
static const int MAXPOOL_OUTPUT_ROWS = 9; // Number of Rows in the output image from Maxpool
static const int MAXPOOL_OUTPUT_COLS = 9;  // Number of Cols in the output image from Maxpool
static const int CONCAT_OUTPUT_ROWS = 9; // Number of Rows in the output image from Concat
static const int CONCAT_OUTPUT_COLS = 9;  // Number of Cols in the output image from Concat
static const int CONCAT_NUMBER_OF_FILTERS = 16;
unsigned char calculatedLabels[NUMBER_OF_IMAGES];// Classified Class after FC
int kernelcalculatedLabels[NUMBER_OF_IMAGES];
unsigned char available_labels[NUMBER_OF_IMAGES]; // Labels from the MNIST Dataset
static const int TOTAL_NUMBER_OF_IMAGE_PIXELS = NUMBER_OF_IMAGES*NUMBER_OF_ROWS*NUMBER_OF_COLS;
static const int TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS = NUMBER_OF_FILTERS*FILTER_ROWS*FILTER_COLS;
static const int TOTAL_NUMBER_OF_CONV_OUT_PIXELS=NUMBER_OF_IMAGES*NUMBER_OF_FILTERS*CONV_OUTPUT_ROWS*CONV_OUTPUT_COLS;
static const int NUMBER_OF_FC_PIXELS =NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS;
static const int NUMBER_OF_FC_WEIGHTS =NUMBER_OF_FC_PIXELS*NUMBER_OF_CLASSES;
static const int NUMBER_OF_PIXELS_FCL = 175 ;

double Kernel_Img[TOTAL_NUMBER_OF_IMAGE_PIXELS] __attribute__ ((aligned (64)));
float Kernel_CNN_WEIGHTS[TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS]  __attribute__ ((aligned (64)));
float Kernel_CNN_BIAS[NUMBER_OF_FILTERS]  __attribute__ ((aligned (64)));
int Kernel_Out[TOTAL_NUMBER_OF_CONV_OUT_PIXELS]  __attribute__ ((aligned (64)));
int CONV_PAD_BEGIN[2] __attribute__ ((aligned (64)));
int CONV_PAD_END[2] __attribute__ ((aligned (64)));

int Conv_Output_local[NUMBER_OF_IMAGES*32*NUMBER_OF_PIXELS] __attribute__ ((aligned (64)));
int Maxpool_Output[NUMBER_OF_IMAGES*DEPTH*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS] __attribute__ ((aligned (64)));





int main(void)
{
CONV_PAD_BEGIN[0] = 1;
CONV_PAD_BEGIN[1] = 1;
CONV_PAD_END[0] = 1;
CONV_PAD_END[1] = 1;

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

	//Create Buffers for input and outputs
        cl::Buffer Buffer_Imgs(mycontext, CL_MEM_READ_ONLY, sizeof(double)* TOTAL_NUMBER_OF_IMAGE_PIXELS);
        cl::Buffer Buffer_ConvWeights(mycontext, CL_MEM_READ_ONLY, sizeof(float)* TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS);
        cl::Buffer Buffer_ConvBias(mycontext, CL_MEM_READ_ONLY, sizeof(float)* NUMBER_OF_FILTERS);
	cl::Buffer Buffer_ConvPadBegin(mycontext, CL_MEM_READ_ONLY, sizeof(int)* 2);
	cl::Buffer Buffer_ConvPadEnd(mycontext, CL_MEM_READ_ONLY, sizeof(int)* 2);	
	cl::Buffer Buffer_ConvOutput(mycontext,CL_MEM_READ_WRITE,sizeof(double)* NUMBER_OF_IMAGES*CONV_OUTPUT_ROWS*CONV_OUTPUT_COLS*NUMBER_OF_FILTERS*DEPTH);

	cl::Buffer Buffer_MaxPoolOutput(mycontext, CL_MEM_READ_WRITE, sizeof(double)* NUMBER_OF_IMAGES*NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS);
	cl::Buffer Buffer_AvgPoolOutput(mycontext, CL_MEM_READ_WRITE, sizeof(double)* NUMBER_OF_IMAGES*NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS);
	//cl::Buffer Buffer_AvgPoolOutput(mycontext, CL_MEM_READ_WRITE, sizeof(int)* NUMBER_OF_PIXELS_FCL * NUMBER_OF_IMAGES);
	//cl::Buffer Buffer_ConcatInput(mycontext, CL_MEM_READ_ONLY, sizeof(int)* NUMBER_OF_PIXELS_FCL * NUMBER_OF_IMAGES * 2);
	//cl::Buffer Buffer_ConcatLayers(mycontext, CL_MEM_READ_ONLY, sizeof(int)* 2);
	cl::Buffer Buffer_ConcatOutput(mycontext, CL_MEM_WRITE_ONLY, sizeof(double)* NUMBER_OF_IMAGES*4*NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS);
	
	//cl::Buffer Buffer_FCLWeights(mycontext, CL_MEM_READ_ONLY, sizeof(short)*NUMBER_OF_FC_WEIGHTS);
	//cl::Buffer Buffer_FCLBias(mycontext, CL_MEM_READ_ONLY, sizeof(short)*NUMBER_OF_CLASSES);
        //cl::Buffer Buffer_FCLOutput(mycontext,CL_MEM_READ_WRITE,sizeof(int)*NUMBER_OF_IMAGES);
	//cl::Buffer Buffer_SoftMaxOutput(mycontext,CL_MEM_READ_ONLY,sizeof(int)*NUMBER_OF_IMAGES);

	
	
	std::cout << "Creating the Images" << std::endl;
	for (int i=0; i < NUMBER_OF_PIXELS*NUMBER_OF_IMAGES; i++) {
		Kernel_Img[i] = 1;	
	}

	std::cout << "Finished creating the Images" << std::endl;

	int temp_count = 0;
/*
	std::cout << "Creating the Class Weights" << std::endl;
        short digitWeights[NUMBER_OF_CLASSES*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS*NUMBER_OF_FILTERS];
	temp_count = 0;
        for(int i=0; i<NUMBER_OF_CLASSES; i++) {
                for(int j=0; j<(CONCAT_OUTPUT_ROWS*CONCAT_OUTPUT_COLS*CONCAT_NUMBER_OF_FILTERS); j++) {
			digitWeights[temp_count] = i;

                }
        }
        std::cout << "Finished creating the Class Weights" << std::endl;
*/

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

	//Write data to device
        err = queueConvLayer.enqueueWriteBuffer(Buffer_Imgs, CL_FALSE, 0, sizeof(double)*TOTAL_NUMBER_OF_IMAGE_PIXELS, Kernel_Img);
        assert(err==CL_SUCCESS);
        err = queueConvLayer.enqueueWriteBuffer(Buffer_ConvWeights, CL_FALSE, 0, sizeof(float)*TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS, Kernel_CNN_WEIGHTS);
        assert(err==CL_SUCCESS);
        err = queueConvLayer.enqueueWriteBuffer(Buffer_ConvBias, CL_FALSE, 0, sizeof(float)*NUMBER_OF_FILTERS, Kernel_CNN_BIAS);
        assert(err==CL_SUCCESS);
        err = queueConvLayer.enqueueWriteBuffer(Buffer_ConvPadBegin, CL_FALSE, 0, sizeof(int)*2, CONV_PAD_BEGIN);
        assert(err==CL_SUCCESS);
        err = queueConvLayer.enqueueWriteBuffer(Buffer_ConvPadEnd, CL_FALSE, 0, sizeof(int)*2, CONV_PAD_END);
        assert(err==CL_SUCCESS);

	// create the kernel
        const char *CONV_kernel_name = "ConvolutionLayer";
        const char *MP_kernel2_name = "MaxPool";
	const char *AP_kernel3_name = "AvgPool";
	const char *Concat_kernel4_name = "ConcatLayer";
        //const char *FC_kernel3_name = "FCL_Kernel";
        //Read in binaries from file
        std::ifstream aocx_stream("./CustomKernels.aocx", std::ios::in|std::ios::binary);
        checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "CustomKernels.aocx");
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

        cl::Kernel kernel3(program,AP_kernel3_name, &err);
        assert(err==CL_SUCCESS);

        cl::Kernel kernel4(program,Concat_kernel4_name, &err);
        assert(err==CL_SUCCESS);

	printf("\nPrint out the image...\n");

	temp_count = 0;
	for (int j = 0; j < NUMBER_OF_IMAGES; j++){
		std::cout << j << " image: " << std::endl;
		for (int i = 0; i < CONV_OUTPUT_ROWS; i++) {
			for (int k = 0; k < CONV_OUTPUT_COLS; k++){
				std::cout << Kernel_Img[temp_count] << " ";
				temp_count++;
			}
			std::cout << std::endl; 	
		}
		std::cout << std::endl;
	}


	printf("\nSetting arguments and launching the ConvKernel...\n");

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
        err = kernel.setArg(7, CONV_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(8, CONV_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(9, DEPTH);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(10, Buffer_ConvPadBegin);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(11, Buffer_ConvPadEnd);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(12, CONV_STRIDE);
        assert(err==CL_SUCCESS);
	err = kernel.setArg(13, Buffer_ConvOutput);
        assert(err==CL_SUCCESS);

	 // Launch Kernel
        err=queueConvLayer.enqueueTask(kernel);
        assert(err==CL_SUCCESS);

	double Conv_Output_local[NUMBER_OF_IMAGES*CONV_OUTPUT_ROWS*CONV_OUTPUT_COLS*NUMBER_OF_FILTERS*DEPTH];
	err=queueConvLayer.enqueueReadBuffer(Buffer_ConvOutput, CL_TRUE, 0, sizeof(double)*(NUMBER_OF_IMAGES*CONV_OUTPUT_ROWS*CONV_OUTPUT_COLS*NUMBER_OF_FILTERS*DEPTH), Conv_Output_local);
        assert(err==CL_SUCCESS);

	printf("\nPrint out the conv image...\n");

	temp_count = 0;
	for (int j = 0; j < NUMBER_OF_IMAGES; j++){
		std::cout << j << " image: " << std::endl;
		for (int l = 0; l < NUMBER_OF_FILTERS; l++){
			std::cout << "# filter: "<< l << std::endl;
			for (int i = 0; i < CONV_OUTPUT_ROWS; i++) {
				for (int k = 0; k < CONV_OUTPUT_COLS; k++){
					//if (l == j)
						std::cout << Conv_Output_local[temp_count] << " ";
					temp_count++;
				}
				//if (l == j)			
					std::cout << std::endl; 	
			}
		}
		std::cout << std::endl;
	}

	err=queueConvLayer.finish();
        assert(err==CL_SUCCESS);


	//err = queueMaxPool.enqueueWriteBuffer(ConvMaxPoolBuffer, CL_FALSE, 0, sizeof(int)*(NUMBER_OF_PIXELS * NUMBER_OF_IMAGES * 32), Conv_Output_local);
        //assert(err==CL_SUCCESS);

	printf("\nSetting arguments and launching the MaxPoolKernel...\n");

	err = kernel2.setArg(0, Buffer_ConvOutput);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(1, CONV_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(2, CONV_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(3, NUMBER_OF_FILTERS);
        assert(err==CL_SUCCESS);
	err = kernel2.setArg(4,CONV_STRIDE);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(5, NUMBER_OF_IMAGES);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(6, Buffer_ConvPadBegin);
        assert(err==CL_SUCCESS);
        err = kernel2.setArg(7, Buffer_ConvPadEnd);
        assert(err==CL_SUCCESS);
	err = kernel2.setArg(8, Buffer_MaxPoolOutput);
        assert(err==CL_SUCCESS);


	err=queueMaxPool.enqueueTask(kernel2);
        assert(err==CL_SUCCESS);

	double MaxPool_Output_local[NUMBER_OF_IMAGES*NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS];
	err=queueMaxPool.enqueueReadBuffer(Buffer_MaxPoolOutput, CL_TRUE, 0, sizeof(double)*(NUMBER_OF_IMAGES*NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS), MaxPool_Output_local);
        assert(err==CL_SUCCESS);

	printf("\nPrint out the MaxPool image...\n");

	temp_count = 0;
	for (int j = 0; j < NUMBER_OF_IMAGES; j++){
		std::cout << j << " image: " << std::endl;
		for (int l = 0; l < NUMBER_OF_FILTERS; l++){
			std::cout << "# filter: "<< l << std::endl;
			for (int i = 0; i < MAXPOOL_OUTPUT_ROWS; i++) {
				for (int k = 0; k < MAXPOOL_OUTPUT_COLS; k++){
					//if (l == j)
						std::cout << MaxPool_Output_local[temp_count] << " ";
					temp_count++;
				}
				//if (l == j)			
					std::cout << std::endl; 	
			}
		}
		std::cout << std::endl;
	}

	err=queueMaxPool.finish();
        assert(err==CL_SUCCESS);



	printf("\nSetting arguments and launching the AvgPoolKernel...\n");

	err = kernel3.setArg(0, Buffer_ConvOutput);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(1, CONV_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(2, CONV_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(3, NUMBER_OF_FILTERS);
        assert(err==CL_SUCCESS);
	err = kernel3.setArg(4,3);
        assert(err==CL_SUCCESS);
	err = kernel3.setArg(5,CONV_STRIDE);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(6, NUMBER_OF_IMAGES);
        assert(err==CL_SUCCESS);
	err = kernel3.setArg(7, Buffer_AvgPoolOutput);
        assert(err==CL_SUCCESS);

	err=queueAvgPool.enqueueTask(kernel);
        assert(err==CL_SUCCESS);

	double AvgPool_Output_local[NUMBER_OF_IMAGES*NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS];
	err=queueAvgPool.enqueueReadBuffer(Buffer_AvgPoolOutput, CL_TRUE, 0, sizeof(double)*(NUMBER_OF_IMAGES*NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS), AvgPool_Output_local);
        assert(err==CL_SUCCESS);

	printf("\nPrint out the AvgPool image...\n");

	temp_count = 0;
	for (int j = 0; j < NUMBER_OF_IMAGES; j++){
		std::cout << j << " image: " << std::endl;
		for (int l = 0; l < NUMBER_OF_FILTERS; l++){
			std::cout << "# filter: "<< l << std::endl;
			for (int i = 0; i < MAXPOOL_OUTPUT_ROWS; i++) {
				for (int k = 0; k < MAXPOOL_OUTPUT_COLS; k++){
					//if (l == j)
						std::cout << AvgPool_Output_local[temp_count] << " ";
					temp_count++;
				}
				//if (l == j)			
					std::cout << std::endl; 	
			}
		}
		std::cout << std::endl;
	}

	err=queueAvgPool.finish();
        assert(err==CL_SUCCESS);

/*
	printf("\nSetting arguments and launching the ConcatKernel...\n");

	err = kernel4.setArg(0, Buffer_MaxPoolOutput);
        assert(err==CL_SUCCESS);
        err = kernel4.setArg(1, Buffer_MaxPoolOutput);
        assert(err==CL_SUCCESS);
        err = kernel4.setArg(2, Buffer_MaxPoolOutput);
        assert(err==CL_SUCCESS);
        err = kernel4.setArg(3, Buffer_MaxPoolOutput);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(4,MAXPOOL_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(5,MAXPOOL_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(6,3);
	assert(err==CL_SUCCESS);
	err = kernel4.setArg(7,MAXPOOL_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(8,MAXPOOL_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(9,3);
	assert(err==CL_SUCCESS);	
	err = kernel4.setArg(10,MAXPOOL_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(11,MAXPOOL_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(12,3);
	assert(err==CL_SUCCESS);
	err = kernel4.setArg(13,MAXPOOL_OUTPUT_ROWS);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(14,MAXPOOL_OUTPUT_COLS);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(15,3);
        assert(err==CL_SUCCESS);
	err = kernel4.setArg(16, Buffer_ConcatOutput);
        assert(err==CL_SUCCESS);


	err=queueConcat.enqueueTask(kernel);
        assert(err==CL_SUCCESS);

	//err=queueConcat.enqueueReadBuffer(maxOutput, CL_TRUE, 0, sizeof(int)*(NUMBER_OF_PIXELS_FCL * NUMBER_OF_IMAGES), Maxpool_Output);
        //assert(err==CL_SUCCESS);

	err=queueConcat.finish();
        assert(err==CL_SUCCESS);

	double Concat_Output[NUMBER_OF_IMAGES*4*DEPTH*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS];
        err = queueConcat.enqueueReadBuffer(Buffer_ConcatOutput, CL_FALSE, 0, sizeof(double)*(NUMBER_OF_IMAGES*4*DEPTH*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS), Concat_Output);
        assert(err==CL_SUCCESS);

*/

}
