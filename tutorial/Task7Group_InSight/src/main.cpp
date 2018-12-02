#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include "CL/cl.hpp"
#include "utility.h"
#include <fstream>
#include <chrono>

static const int NUMBER_OF_IMAGES  = 10000 ;
static const int NUMBER_OF_PIXELS = 784 ;
static const int NUMBER_OF_CLASSES = 10 ; //0 to 9
static const int NUMBER_OF_FILTERS = 32 ;
static const int NUMBER_OF_ROWS = 32; //Including zero padding
static const int NUMBER_OF_COLS = 32; //Including zero padding
static const int FILTER_ROWS = 5;
static const int FILTER_COLS = 5;
static const int ZERO_PADDING = 2;
static const int CONV_LAYER_OUTPUT_ROWS = 28;
static const int CONV_LAYER_OUTPUT_COLS = 28;

int main(void)
{


	//Read Input Data from MNIST Database and store it in a 3D vector . ImageReader[NumberOfImages][NumberOfRows][NumberOfCols]
	std::vector<std::vector<std::vector<unsigned char>>> ImageReader;
	ReadMNIST_char(NUMBER_OF_IMAGES,NUMBER_OF_ROWS,NUMBER_OF_COLS,ZERO_PADDING,ImageReader);

    std::cout << "Sample Image Pixel Value:"  << std::endl;
      for(int i=0;i<NUMBER_OF_ROWS;i++){
        for(int j=0;j<NUMBER_OF_COLS;j++){
      std::cout << (int)ImageReader[0][i][j] << " ";
      }
      std::cout<< std::endl;
    }

    std::cout << "Finished Reading the MNIST Images" << std::endl;

    short Weights_2D[NUMBER_OF_CLASSES][NUMBER_OF_PIXELS];
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


      int CNN_Weights_1D[((FILTER_ROWS*FILTER_COLS)+1)*NUMBER_OF_FILTERS];
      std::vector<std::vector<std::vector<short>>> CNNWeights;
      std::vector<short> cnnbias;
      char path_to_cnn_weight[1024] = { "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/cnn_weights"};
      read_cnn_weights_file_char(path_to_cnn_weight, CNNWeights,cnnbias,FILTER_ROWS,FILTER_COLS,NUMBER_OF_FILTERS);

      std::cout << "Sample Conv Filter Weights" << std::endl;
      for(int i=0;i<5;i++){
        for(int j=0;j<5;j++){
          std::cout << CNNWeights[2][i][j]<< " ";
        }
        std::cout << std::endl;
      }
      std::cout << "bias :"<<cnnbias[2] << std::endl;
      std::cout << "Finished Reading the CNN Weights" << std::endl;


      std::vector<std::vector<std::vector<long>>> ConvOutput;
      for(int i=0;i<1;i++)
        for(int j=0;j<NUMBER_OF_FILTERS;j++)
          convlutionLayer(ImageReader[i],CNNWeights[j],cnnbias[j],FILTER_ROWS,FILTER_COLS,NUMBER_OF_ROWS,NUMBER_OF_COLS,ConvOutput[j],CONV_LAYER_OUTPUT_ROWS,CONV_LAYER_OUTPUT_COLS);

      std::cout << "Finished Convolution" << std::endl;
	printf("\nDone.\n");

	return 1;


}
