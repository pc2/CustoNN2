#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include "CL/cl.hpp"
#include "utility.h"
#include <fstream>
#include <chrono>

static const int NUMBER_OF_IMAGES  = 10000;
static const int NUMBER_OF_PIXELS = 784;
static const int NUMBER_OF_CLASSES = 10;  //0 to 9
static const int NUMBER_OF_FILTERS = 32;
static const int NUMBER_OF_ROWS = 32; //Including zero padding
static const int NUMBER_OF_COLS = 32; //Including zero padding
static const int FILTER_ROWS = 5; // Number of rows in the conv Filter
static const int FILTER_COLS = 5; // Number of rows in the conv Filter
static const int ZERO_PADDING = 2; // Number of Zero Padding
static const int STRIDE=2; // Stride
static const int CONV_LAYER_OUTPUT_ROWS = 28; // NUmber of Rows in the Output image from Conv Layer
static const int CONV_LAYER_OUTPUT_COLS = 28; // NUmber of Cols in the Output image from Conv Layer
static const int MAXPOOL_OUTPUT_ROWS = 14; // Number of Rows in the output image from Maxpool
static const int MAXPOOL_OUTPUT_COLS = 14;  // Number of Cols in the output image from Maxpool
unsigned char calculatedLabels[NUMBER_OF_IMAGES];// Classified Class after FC
int kernelcalculatedLabels[NUMBER_OF_IMAGES];
unsigned char available_labels[NUMBER_OF_IMAGES]; // Labels from the MNIST Dataset
static const int TOTAL_NUMBER_OF_IMAGE_PIXELS = NUMBER_OF_IMAGES*NUMBER_OF_ROWS*NUMBER_OF_COLS;
static const int TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS = NUMBER_OF_FILTERS*FILTER_ROWS*FILTER_COLS;
static const int TOTAL_NUMBER_OF_CONV_OUT_PIXELS=NUMBER_OF_IMAGES*NUMBER_OF_FILTERS*CONV_LAYER_OUTPUT_ROWS*CONV_LAYER_OUTPUT_COLS;
static const int NUMBER_OF_FC_PIXELS =NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS;
static const int NUMBER_OF_FC_WEIGHTS =NUMBER_OF_FC_PIXELS*NUMBER_OF_CLASSES;

char Kernel_Img[TOTAL_NUMBER_OF_IMAGE_PIXELS] __attribute__((aligned (64)));
short Kernel_CNN_WEIGHTS[TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS]  __attribute__((aligned (64)));
short Kernel_CNN_BIAS[NUMBER_OF_FILTERS]  __attribute__((aligned (64)));
int Kernel_Out[TOTAL_NUMBER_OF_CONV_OUT_PIXELS]  __attribute__((aligned (64)));



int main(void)
{
        // FPGA Implementation
        cl_int err;

        //Setup Platform
        std::cout << "Setup Platform" << std::endl;
        //Get Platform ID
        std::vector<cl::Platform> PlatformList;
        ////////////// Exercise 1 Step 2.3
        err = cl::Platform::get(&PlatformList);
        assert(err==CL_SUCCESS);
        checkErr(PlatformList.size()==1 ? CL_SUCCESS : -1, "cl::Platform::get");
        print_platform_info(&PlatformList);
        std::cout << "Done Initializing Platform" << std::endl;
        //Setup Device
        //Get Device ID
        std::vector<cl::Device> DeviceList,Device1,Device2;
        err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList);
        assert(err==CL_SUCCESS);
        print_device_info(&DeviceList);

        //Copy the Devices into another List having only 1 device.
        Device1.push_back(DeviceList[0]); 
        Device2.push_back(DeviceList[1]);
       /* std::vector<cl::Device> DeviceList2;
        err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList2);
        print_device_info(&DeviceList2);
        */

        //Create Context
        cl::Context mycontext(Device1);
        assert(err==CL_SUCCESS);

        cl::Context mycontext2(Device2);
        assert(err==CL_SUCCESS);

        //Create Command queue
        cl::CommandQueue queueConvLayer(mycontext, Device1[0]);
        assert(err==CL_SUCCESS);
        cl::CommandQueue queueMaxPool(mycontext, Device1[0]);
        assert(err==CL_SUCCESS);
        cl::CommandQueue queueFCLayer(mycontext2, Device2[0]);
        assert(err==CL_SUCCESS);

        //Create Buffers for input and output
        cl::Buffer Buffer_Img(mycontext, CL_MEM_READ_ONLY, sizeof(char)*TOTAL_NUMBER_OF_IMAGE_PIXELS);
        cl::Buffer Buffer_CNN_WEIGHTS(mycontext, CL_MEM_READ_ONLY, sizeof(short)*TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS);
        cl::Buffer Buffer_CNN_BIAS(mycontext, CL_MEM_READ_ONLY, sizeof(short)*NUMBER_OF_FILTERS);
        cl::Buffer Buffer_digitWeights(mycontext2, CL_MEM_READ_ONLY, sizeof(short)*NUMBER_OF_FC_WEIGHTS);
        cl::Buffer Buffer_kernelcalculatedLabels(mycontext2,CL_MEM_WRITE_ONLY,sizeof(int)*NUMBER_OF_IMAGES);
     

        //Inputs and Outputs to Kernel, X and Y are inputs, Z is output
        //The aligned attribute is used to ensure alignment
        //so that DMA could be used if we were working with a real FPGA board

        std::cout << "Reading 10k MNIST Dataset Images" << std::endl;
        //Read Input Data from MNIST Database and store it in a 3D vector . ImageReader[NumberOfImages][NumberOfRows][NumberOfCols]
        std::vector<std::vector<std::vector<unsigned char> > > ImageReader;
        ReadMNIST_char(NUMBER_OF_IMAGES,NUMBER_OF_ROWS,NUMBER_OF_COLS,ZERO_PADDING,ImageReader);


        //Convert 3D vector into 1 D Array - For Kernel
        for(int i=0; i<NUMBER_OF_IMAGES; i++) {
                for(int j=0; j<NUMBER_OF_ROWS; j++) {
                        for(int k=0; k<NUMBER_OF_COLS; k++) {
                                Kernel_Img[(i*NUMBER_OF_ROWS*NUMBER_OF_ROWS)+(j*NUMBER_OF_ROWS)+k] = ImageReader[i][j][k];
                        }
                }
        }

        /*
           std::cout << "Sample Image Pixel Value:"  << std::endl;
           for(int i=0; i<NUMBER_OF_ROWS; i++) {
                for(int j=0; j<NUMBER_OF_COLS; j++) {
                        std::cout << (int)ImageReader[0][i][j] << " ";
                }
                std::cout<< std::endl;
           }
         */
        std::cout << "Finished Reading the MNIST Images" << std::endl;

        std::cout << "Reading MNIST Dataset Weights" << std::endl;
        short Weights_2D[NUMBER_OF_CLASSES][MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS*NUMBER_OF_FILTERS];
        char path_to_file_0[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_0"};
        char path_to_file_1[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_1"};
        char path_to_file_2[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_2"};
        char path_to_file_3[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_3"};
        char path_to_file_4[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_4"};
        char path_to_file_5[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_5"};
        char path_to_file_6[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_6"};
        char path_to_file_7[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_7"};
        char path_to_file_8[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_8"};
        char path_to_file_9[1024] = {"/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/fc_weights_9"};

        // Call  the function given in the manual
        read_weights_file_char(path_to_file_0, Weights_2D[0]);
        read_weights_file_char(path_to_file_1, Weights_2D[1]);
        read_weights_file_char(path_to_file_2, Weights_2D[2]);
        read_weights_file_char(path_to_file_3, Weights_2D[3]);
        read_weights_file_char(path_to_file_4, Weights_2D[4]);
        read_weights_file_char(path_to_file_5, Weights_2D[5]);
        read_weights_file_char(path_to_file_6, Weights_2D[6]);
        read_weights_file_char(path_to_file_7, Weights_2D[7]);
        read_weights_file_char(path_to_file_8, Weights_2D[8]);
        read_weights_file_char(path_to_file_9, Weights_2D[9]);

        std::cout << "Finished Reading the Class Weights" << std::endl;

        //Read labels given in the shared location
        read_labels_file(available_labels);

        std::vector<std::vector<std::vector<short> > > CNNWeights;
        std::vector<short> cnnbias;
        char path_to_cnn_weight[1024] = { "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/cnn_weights"};
        read_cnn_weights_file_char(path_to_cnn_weight, CNNWeights,cnnbias,FILTER_ROWS,FILTER_COLS,NUMBER_OF_FILTERS);
        std::cout << "Finished Reading the CNN Weights" << std::endl;

        // Convert 3D CNN Weights Vector to  1D array
        for(int i=0; i<NUMBER_OF_FILTERS; i++) {
                for(int j=0; j<FILTER_ROWS; j++) {
                        for(int k=0; k<FILTER_COLS; k++) {
                                Kernel_CNN_WEIGHTS[(i*FILTER_ROWS*FILTER_COLS)+(j*FILTER_ROWS)+k]= CNNWeights[i][j][k];
                        }
                }
        }
        //Bias Array
        for(int i=0; i<NUMBER_OF_FILTERS; i++)
                Kernel_CNN_BIAS[i]=cnnbias[i];



        short digitWeights[NUMBER_OF_CLASSES*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS*NUMBER_OF_FILTERS];
        // Convert 2D digit Weights Vector to  1D array
        for(int i=0; i<NUMBER_OF_CLASSES; i++) {
                for(int j=0; j<(MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS*NUMBER_OF_FILTERS); j++) {

                        digitWeights[(i*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS*NUMBER_OF_FILTERS)+j]= Weights_2D[i][j];

                }
        }


        /*
           std::cout << "Sample Conv Filter Weights" << std::endl;

           for(int i=0; i<5; i++) {
                for(int j=0; j<5; j++) {
                        std::cout << CNNWeights[0][i][j]<< " ";
                }
                std::cout << std::endl;
           }
           std::cout << "bias :"<<cnnbias[0] << std::endl;
         */


        //Write data to device
        err = queueConvLayer.enqueueWriteBuffer(Buffer_Img, CL_FALSE, 0, sizeof(char)*TOTAL_NUMBER_OF_IMAGE_PIXELS, Kernel_Img);

        assert(err==CL_SUCCESS);
        err = queueConvLayer.enqueueWriteBuffer(Buffer_CNN_WEIGHTS, CL_FALSE, 0, sizeof(short)*TOTAL_NUMBER_OF_CNN_WEIGHT_PIXELS, Kernel_CNN_WEIGHTS);
        assert(err==CL_SUCCESS);
        err = queueConvLayer.enqueueWriteBuffer(Buffer_CNN_BIAS, CL_FALSE, 0, sizeof(short)*NUMBER_OF_FILTERS, Kernel_CNN_BIAS);
        assert(err==CL_SUCCESS);
        err = queueFCLayer.enqueueWriteBuffer(Buffer_digitWeights, CL_FALSE, 0, sizeof(short)*NUMBER_OF_FC_WEIGHTS, digitWeights);
        assert(err==CL_SUCCESS);

        // create the kernel
        const char *CONV_kernel_name = "ConvLayer";
        const char *MP_kernel2_name = "MaxPool";
        const char *FC_kernel3_name = "FCLayer";
        //Read in binaries from file
        std::ifstream aocx_stream("../Simple_ConvolutionNeuralNetwork.aocx", std::ios::in|std::ios::binary);
        checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionalNeuralNetwork.aocx");
        std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
        cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()+1));


        std::ifstream aocx_stream2("../Simple_ConvolutionNeuralNetwork.aocx", std::ios::in|std::ios::binary);
        checkErr(aocx_stream.is_open() ? CL_SUCCESS : -1, "Simple_ConvolutionNeuralNetwork.aocx");
        std::string prog2(std::istreambuf_iterator<char>(aocx_stream2), (std::istreambuf_iterator<char>()));
        cl::Program::Binaries mybinaries2 (1, std::make_pair(prog2.c_str(), prog2.length()+1));


        // Create the Program from the AOCX file.
        cl::Program program(mycontext, Device1, mybinaries);

        // Create the Program from the AOCX file.
        cl::Program program2(mycontext2, Device2, mybinaries2);

        // build the program
        err=program.build(Device1);
        std::cout << err<<std::endl;
        assert(err==CL_SUCCESS);

        err=program2.build(Device2);
        std::cout << err<<std::endl;
        assert(err==CL_SUCCESS);

        // create the kernel
        cl::Kernel kernel(program, CONV_kernel_name, &err);
        assert(err==CL_SUCCESS);

        cl::Kernel kernel2(program,MP_kernel2_name, &err);
        assert(err==CL_SUCCESS);

        cl::Kernel kernel3(program2,FC_kernel3_name, &err);
        assert(err==CL_SUCCESS);


        //////////////     Set Arguments to the Kernels
        err = kernel.setArg(0, Buffer_Img);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(1, Buffer_CNN_WEIGHTS);
        assert(err==CL_SUCCESS);
        err = kernel.setArg(2, Buffer_CNN_BIAS);
        assert(err==CL_SUCCESS);



        err = kernel3.setArg(0, Buffer_digitWeights);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(1, NUMBER_OF_FC_PIXELS);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(2, NUMBER_OF_CLASSES);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(3, NUMBER_OF_IMAGES);
        assert(err==CL_SUCCESS);
        err = kernel3.setArg(4, Buffer_kernelcalculatedLabels);
        assert(err==CL_SUCCESS);

        printf("\nLaunching the kernels...\n");

        auto startFPGA = std::chrono::high_resolution_clock::now();
        // Launch Kernel
        err=queueConvLayer.enqueueTask(kernel);
        assert(err==CL_SUCCESS);
        //err=queueConvLayer.finish();
        err=queueMaxPool.enqueueTask(kernel2);
        assert(err==CL_SUCCESS);
        //err=queueMaxPool.finish();
        err=queueFCLayer.enqueueTask(kernel3);
        assert(err==CL_SUCCESS);

        // read the output
        err=queueFCLayer.enqueueReadBuffer(Buffer_kernelcalculatedLabels, CL_TRUE, 0, sizeof(int)*NUMBER_OF_IMAGES, kernelcalculatedLabels);
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

        std::cout << "Starting Convolution for 10k images and 32 filters..." << std::endl;
        std::vector<std::vector<int> > ConvOutput;
        std::vector<std::vector<std::vector<int> > > ConvOutputFilters(NUMBER_OF_FILTERS,std::vector<std::vector<int> >(CONV_LAYER_OUTPUT_ROWS,std::vector <int>(CONV_LAYER_OUTPUT_COLS)));
        std::vector<std::vector<std::vector<int> > > MaxPoolOutput(NUMBER_OF_FILTERS,std::vector<std::vector<int> >(MAXPOOL_OUTPUT_ROWS,std::vector <int>(MAXPOOL_OUTPUT_COLS)));


        //Start Time
        auto startTime = std::chrono::high_resolution_clock::now();

        //Main Computation
        for(int i=0; i<NUMBER_OF_IMAGES; i++) {
                for(int j=0; j<NUMBER_OF_FILTERS; j++) {


                        /*
                         * Convolution Layer ( activation function: ReLU)
                         * input : 32*28*28 Image + 2 Zero Padding, 32 5*5 Conv Filter , 1 bias for each filter,1 stride
                         * Output : 32*28*28 Convoluted Image.
                         */
                        convlutionLayer(ImageReader[i],CNNWeights[j],cnnbias[j],FILTER_ROWS,FILTER_COLS,NUMBER_OF_ROWS,NUMBER_OF_COLS,ConvOutput,CONV_LAYER_OUTPUT_ROWS,CONV_LAYER_OUTPUT_COLS);
                        //  std::cout << "Finished Convolution for image :"<<i <<" and Filter : "<<j << std::endl;

                        // form 32 filter outputs of conv layer.
                        for(int k=0; k<CONV_LAYER_OUTPUT_ROWS; k++)
                                for(int l=0; l<CONV_LAYER_OUTPUT_COLS; l++)
                                        ConvOutputFilters[j][k][l]=ConvOutput[k][l];
                        /*
                           if(i==0) {
                                  std::cout << "Test Convoluted result :"<<j << std::endl;
                                  for(int k=0; k<CONV_LAYER_OUTPUT_ROWS; k++) {
                                          for(int l=0; l<CONV_LAYER_OUTPUT_COLS; l++) {
                                                  std::cout << ConvOutputFilters[j][k][l]<< " ";
                                          }
                                          std::cout << std::endl;
                                  }
                           }
                         */
                }




                /*
                 * Max Pool Layer
                 * input : 32*28*28 Image
                 * Output : 32*14*14 Image. This image will be converted to 1D of 6272 elements
                 */
                //  std::cout << "Starting Maxpool for image :"<<i<<" and 32 filters..." << std::endl;

                maxpoolLayer(ConvOutputFilters,MaxPoolOutput,NUMBER_OF_FILTERS,CONV_LAYER_OUTPUT_ROWS,CONV_LAYER_OUTPUT_COLS,STRIDE);
                //  std::cout << "Finished Maxpool" << std::endl;
                /*    std::cout << "Test MaxPool result" << std::endl;
                    for(int k=0;k<MAXPOOL_OUTPUT_ROWS;k++){
                      for(int l=0;l<MAXPOOL_OUTPUT_COLS;l++){
                        std::cout << MaxPoolOutput[0][k][l]<< "\t";
                      }
                      std::cout << std::endl;
                    } */

                //convert 2D Maxpool Output to 1D of 32*14*14 elements
                int MaxPoolOutput_1D[NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS];
                int poolIndex= 0;
                for(int m=0; m<NUMBER_OF_FILTERS; m++) {
                        for(int n=0; n<MAXPOOL_OUTPUT_ROWS; n++) {
                                for(int p=0; p<MAXPOOL_OUTPUT_COLS; p++) {
                                        MaxPoolOutput_1D[poolIndex] = MaxPoolOutput[m][n][p];
                                        poolIndex++;
                                }
                        }
                }

                /*
                 * Fully Connected Layer
                 * input : Image[6272] and 10  Weights[6272]
                 * Output : Neuron having max score.
                 */
                int maxScore=0;
                int neuron=0;
                int score=0;
                int NUMBER_OF_FC_PIXELS =NUMBER_OF_FILTERS*MAXPOOL_OUTPUT_ROWS*MAXPOOL_OUTPUT_COLS;
                for(int weightIndex=0; weightIndex<NUMBER_OF_CLASSES; weightIndex++) {
                        score=fullyConnectedLayer(MaxPoolOutput_1D,Weights_2D[weightIndex],NUMBER_OF_FC_PIXELS);
                        if(score>maxScore) {
                                maxScore=score;
                                neuron=weightIndex;
                        }
                        //  std::cout << "score "<<weightIndex<<" :"<< score << '\n';
                }
                calculatedLabels[i]=neuron;

        } //end of CNN
        std::cout << "Finished Convolution of 10k images and 32 filters..." << std::endl;
        auto endNew = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endNew - startTime;
        std::cout << "CPU Time Taken for Convolution of 10k images and 32 filters (in sec) :" <<elapsed.count()<< std::endl;


        //Accuracy Calculation
        float counter = 0;
        for(int zc = 0; zc < NUMBER_OF_IMAGES; zc++)
        {
                if(calculatedLabels[zc] == (int)available_labels[zc])
                        counter++;

                //if(zc<100)
                //std::cout << "CPU Value :" <<calculatedLabels[zc] << " ,Label"<<(int)available_labels[zc] << '\n';
        }
        std::cout << "Number of Images correctly classified: " << counter <<std::endl;
        float Accuracy = (counter/ NUMBER_OF_IMAGES) * 100;

        printf("CPU Accuracy is %f\n",Accuracy);

        printf("\nCPU Computation Done.\n");
        /*
           std::cout << "Test Convoluted result" << std::endl;
           for(int k=0;k<CONV_LAYER_OUTPUT_ROWS;k++){
           for(int l=0;l<CONV_LAYER_OUTPUT_COLS;l++){
            std::cout << ConvOutputFilters[0][k][l]<< "\t";
           }
           std::cout << std::endl;
           }

           std::cout << "Test MaxPool result" << std::endl;
           for(int k=0;k<MAXPOOL_OUTPUT_ROWS;k++){
           for(int l=0;l<MAXPOOL_OUTPUT_COLS;l++){
            std::cout << MaxPoolOutput[0][k][l]<< "\t";
           }
           std::cout << std::endl;
           }

         */

        return 0;


}
