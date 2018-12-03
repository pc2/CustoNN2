#ifndef __UTILITY_H
#define __UTILITY_H

#include "CL/cl.hpp"
#include <assert.h>


#define EPSILON (1e-2f)
void print_platform_info(std::vector<cl::Platform>*);
void print_device_info(std::vector<cl::Device>*);
void checkErr(cl_int err, const char * name);


cl::Context Platform_Device_Context_Setup(std::vector<cl::Device> &DeviceList);

void ReadMNIST(int NumberOfImages,int NumberOfPixels, std::vector<std::vector<double>> &arr);
int ReverseInt (int i);

bool read_weights_file(char* filename , float *weights);
void read_labels_file(unsigned char *available_labels);

// Function for Linear Classification Algorithm in CPU
void linearClassifier(float *img,float *weight,float *score,int numberOfImages,int numberOfPixels,int classes);

void linearClassifier_fxp(unsigned char *img,int *weight,int *score,int numberOfImages,int numberOfPixels,int classes);

bool read_weights_file_char(char *filename , short *weights);
bool read_cnn_weights_file_char(char *filename , std::vector<std::vector<std::vector<short>>> &CNNWeights,std::vector<short> &cnnbias,int FILTER_ROWS,int FILTER_COLS,int NUMBER_OF_FILTERS);
void ReadMNIST_char(int NumberOfImages,int NumberOfRows,int NumberOfCols,int ZERO_PADDING,std::vector<std::vector<std::vector<unsigned char>>> &arr);

void convlutionLayer(std::vector<std::vector<unsigned char>> &ImageReader,std::vector<std::vector<short>> &CNNWeights,short cnnbias,int FILTER_ROWS,int FILTER_COLS,
  int NUMBER_OF_ROWS,int NUMBER_OF_COLS,std::vector<std::vector<long>> &ConvOutput, int CONV_LAYER_OUTPUT_ROWS, int CONV_LAYER_OUTPUT_COLS);

void maxpoolLayer(std::vector<std::vector<std::vector<long>>> &ConvOutputFilters,std::vector<std::vector<std::vector<long>>> &MaxPoolOutput,int NUMBER_OF_FILTERS,int NUMBER_OF_ROWS,int NUMBER_OF_COLS,int STRIDE);
#endif
