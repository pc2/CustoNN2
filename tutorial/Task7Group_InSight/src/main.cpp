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
static const int CONV_LAYER_OUTPUT_ROWS = 28; // NUmber of Rows in the Output image from Conv Layer
static const int CONV_LAYER_OUTPUT_COLS = 28; // NUmber of Cols in the Output image from Conv Layer
static const int MAXPOOL_OUTPUT_ROWS = 14; // Number of Rows in the output image from Maxpool
static const int MAXPOOL_OUTPUT_COLS = 14;  // Number of Cols in the output image from Maxpool
unsigned char calculatedLabels[NUMBER_OF_IMAGES]; // Classified Class after FC
unsigned char available_labels[NUMBER_OF_IMAGES]; // Labels from the MNIST Dataset
int main(void)
{

        std::cout << "Reading 10k MNIST Dataset Images" << std::endl;
        //Read Input Data from MNIST Database and store it in a 3D vector . ImageReader[NumberOfImages][NumberOfRows][NumberOfCols]
        std::vector<std::vector<std::vector<unsigned char> > > ImageReader;
        ReadMNIST_char(NUMBER_OF_IMAGES,NUMBER_OF_ROWS,NUMBER_OF_COLS,ZERO_PADDING,ImageReader);


        std::cout << "Sample Image Pixel Value:"  << std::endl;
        for(int i=0; i<NUMBER_OF_ROWS; i++) {
                for(int j=0; j<NUMBER_OF_COLS; j++) {
                        std::cout << (int)ImageReader[0][i][j] << " ";
                }
                std::cout<< std::endl;
        }

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



        std::vector<std::vector<std::vector<short> > > CNNWeights;
        std::vector<short> cnnbias;
        char path_to_cnn_weight[1024] = { "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/weights_fxp/cnn_weights"};
        read_cnn_weights_file_char(path_to_cnn_weight, CNNWeights,cnnbias,FILTER_ROWS,FILTER_COLS,NUMBER_OF_FILTERS);
        std::cout << "Finished Reading the CNN Weights" << std::endl;

        //Read labels given in the shared location
        read_labels_file(available_labels);



        std::cout << "Sample Conv Filter Weights" << std::endl;

        for(int i=0; i<5; i++) {
                for(int j=0; j<5; j++) {
                        std::cout << CNNWeights[0][i][j]<< " ";
                }
                std::cout << std::endl;
        }
        std::cout << "bias :"<<cnnbias[0] << std::endl;


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
                }
                /*      std::cout << "Test Convoluted result" << std::endl;
                      for(int k=0;k<CONV_LAYER_OUTPUT_ROWS;k++){
                        for(int l=0;l<CONV_LAYER_OUTPUT_COLS;l++){
                          std::cout << ConvOutputFilters[0][k][l]<< "\t";
                        }
                        std::cout << std::endl;
                      } */



                /*
                 * Max Pool Layer
                 * input : 32*28*28 Image
                 * Output : 32*14*14 Image. This image will be converted to 1D of 6272 elements
                 */
                //  std::cout << "Starting Maxpool for image :"<<i<<" and 32 filters..." << std::endl;
                int STRIDE=2;
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
        std::cout << "Time Taken for Convolution of 10k images and 32 filters (in sec) :" <<elapsed.count()<< std::endl;


        //Accuracy Calculation
        float counter = 0;
        for(int zc = 0; zc < NUMBER_OF_IMAGES; zc++)
        {
                if(calculatedLabels[zc] == available_labels[zc])
                        //std::cout << "Label:" <<available_labels[0]<< '\n';
                        counter++;
        }
        std::cout << "Number of Images correctly classified: " << counter <<std::endl;
        float Accuracy = (counter/ NUMBER_OF_IMAGES) * 100;

        printf("Accuracy is %f\n",Accuracy);

        printf("\nDone.\n");
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
