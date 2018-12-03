// This file
#include "utility.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <fstream>

#define LABEL_PATH "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/t10k-labels.idx1-ubyte"
#define IMAGE_DATASET_PATH "/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/datasets/Tutorial_Task7_MNIST_files/t10k-images.idx3-ubyte"

void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                 << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
   }
}


void print_platform_info(std::vector<cl::Platform>* PlatformList)
{
	//Grab Platform Info for each platform
	for (int i=0; i<PlatformList->size(); i++)
	{
		printf("Platform Number: %d\n", i);
		std::cout << "Platform Name: "<<PlatformList->at(i).getInfo<CL_PLATFORM_NAME>()<<"\n";
		std::cout << "Platform Profile: "<<PlatformList->at(i).getInfo<CL_PLATFORM_PROFILE>()<<"\n";
		std::cout << "Platform Version: "<<PlatformList->at(i).getInfo<CL_PLATFORM_VERSION>()<<"\n";
		std::cout << "Platform Vendor: "<<PlatformList->at(i).getInfo<CL_PLATFORM_VENDOR>()<<"\n\n";
	}
}


void print_device_info(std::vector<cl::Device>* DeviceList)
{
	//Grab Device Info for each device
	for (int i=0; i<DeviceList->size(); i++)
	{
		printf("Device Number: %d\n", i);
		std::cout << "Device Name: "<<DeviceList->at(i).getInfo<CL_DEVICE_NAME>()<<"\n";
		std::cout << "Device Vendor: "<<DeviceList->at(i).getInfo<CL_DEVICE_VENDOR>()<<"\n";
		std::cout << "Is Device Available?: "<<DeviceList->at(i).getInfo<CL_DEVICE_AVAILABLE>()<<"\n";
		std::cout << "Is Device Little Endian?: "<<DeviceList->at(i).getInfo<CL_DEVICE_ENDIAN_LITTLE>()<<"\n";
		std::cout << "Device Max Compute Units: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()<<"\n";
		std::cout << "Device Max Work Item Dimensions: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()<<"\n";
		std::cout << "Device Max Work Group Size: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()<<"\n";
		std::cout << "Device Max Frequency: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()<<"\n";
		std::cout << "Device Max Mem Alloc Size: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()<<"\n\n";
	}
}

//Function to query and setup platform device and context
cl::Context Platform_Device_Context_Setup(std::vector<cl::Device> &DeviceList)
{
	//Get Platform
	cl_int err;
	std::vector<cl::Platform> PlatformList;
	err = cl::Platform::get(&PlatformList);
	assert(err==CL_SUCCESS);
	checkErr(PlatformList.size() >= 1 ? CL_SUCCESS : -1, "cl::Platform::get");
	print_platform_info(&PlatformList);

	//Get Device ID
	err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList);
	assert(err==CL_SUCCESS);
	print_device_info(&DeviceList);

	 //Create Context
	cl::Context mycontext(DeviceList);
	assert(err==CL_SUCCESS);


	return mycontext;

}


int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}


void ReadMNIST(int NumberOfImages,int NumberOfPixels, std::vector<std::vector<double>> &arr)
{
    arr.resize(NumberOfImages,std::vector<double>(NumberOfPixels));
    std::ifstream file (IMAGE_DATASET_PATH,std::ios::binary);

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number=(int)magic_number;
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images=(int)number_of_images;
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows=(int)n_rows;
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols=(int)n_cols;
        n_cols= ReverseInt(n_cols);

        assert(magic_number== 2051);

        std::cout << " Magic number is " << magic_number << std::endl ;
        std::cout << " Number of Images is " << number_of_images << std::endl ;
        std::cout << " Number of rows is " << n_rows << std::endl ;
        std::cout << " Number of coloumns  is " << n_cols << std::endl ;




        for(int i=0;i<number_of_images;++i)
        	{
        		for(int r=0;r<n_rows;++r)
        		{
        			for(int c=0;c<n_cols;++c)
        			{
        				unsigned char temp=0;
        				file.read((char*)&temp,sizeof(temp));
        				arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }

}



void ReadMNIST_char(int NumberOfImages,int NumberOfRows,int NumberOfCols,int ZERO_PADDING,std::vector<std::vector<std::vector<unsigned char>>> &arr)
{
    int NumberOfPixels=NumberOfRows*NumberOfCols;
    //arr.resize(NumberOfImages,std::vector<std::vector<unsigned char>>(NumberOfPixels));
    std::ifstream file (IMAGE_DATASET_PATH,std::ios::binary);

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number=(int)magic_number;
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images=(int)number_of_images;
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows=(int)n_rows;
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols=(int)n_cols;
        n_cols= ReverseInt(n_cols);

        assert(magic_number== 2051);

        std::cout << " Magic number is " << magic_number << std::endl ;
        std::cout << " Number of Images is " << number_of_images << std::endl ;
        std::cout << " Number of rows is " << n_rows << std::endl ;
        std::cout << " Number of columns  is " << n_cols << std::endl ;



        arr.resize(number_of_images);
        for(int i=0;i<number_of_images;++i)
        	{
            arr[i].resize(n_rows+(2*ZERO_PADDING));
        		for(int r=0;r<(n_rows+(2*ZERO_PADDING));++r)
        		{
              arr[i][r].resize(n_cols+(2*ZERO_PADDING));
        			for(int c=0;c<(n_cols+(2*ZERO_PADDING));++c)
        			{
                // ZERO PADDING
                if(r== 0 || r==1 || r == NumberOfRows-1 || r == NumberOfRows-2 || c== 0 || c==1 || c == NumberOfCols-1 || c == NumberOfCols-2 ){
                  arr[i][r][c] = 0;
                  continue;
                }

        				unsigned char temp=0;
        				file.read((char*)&temp,sizeof(temp));
        				arr[i][r][c]= (unsigned char)temp;
                }
            }
        }
        std::cout << "Done inserting" << std::endl;
    }

}





bool read_weights_file(char *filename , float *weights)
{
	    int NUMBER_OF_PIXELS = 784 ;
		FILE *f = fopen(filename, "r");
		if (f == NULL){
			printf("ERROR: Could not open %s\n", filename);
			return false;
		}
		int read_elements = fread(weights, sizeof(float), NUMBER_OF_PIXELS, f);
		fclose(f);

		if (read_elements != NUMBER_OF_PIXELS){
			printf("ERROR: Read incorrect number of weights from %s\n", filename);
			return false;
		}
		return true;
}


bool read_weights_file_char(char *filename , short *weights)
{
	    int NUMBER_OF_PIXELS = 784 ;
		FILE *f = fopen(filename, "r");
		if (f == NULL){
			printf("ERROR: Could not open %s\n", filename);
			return false;
		}
		int read_elements = fread(weights, sizeof(short), NUMBER_OF_PIXELS, f);
		fclose(f);

		if (read_elements != NUMBER_OF_PIXELS){
			printf("ERROR: Read incorrect number of weights from %s\n", filename);
			return false;
		}
		return true;
}


void read_labels_file(unsigned char *available_labels)
{
	std::ifstream file (LABEL_PATH, std::ios::binary);
	if(file.is_open())
	{
		int magic_number=0;
		int No_Of_Images=0;

		file.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);

		file.read((char*)&No_Of_Images,sizeof(No_Of_Images));
		No_Of_Images=ReverseInt(No_Of_Images);

	    assert(magic_number== 2049);
		std::cout << "Magic number of labels is " << magic_number << std::endl ;

		assert(No_Of_Images== 10000);
		std::cout << "Number of Images in labels file is  " << No_Of_Images << std::endl ;

		for(int i=0;i<No_Of_Images;i++)
		{
		 unsigned char temp=0;
		 file.read((char*)&temp,sizeof(temp));
		 available_labels[i]= temp;
		}
	}

	}


/*
 * Function for Linear Classification Algorithm in CPU
 */
void linearClassifier(float *img,float *weight,float *score,int numberOfImages,int numberOfPixels,int classes){
	  for(int i=0; i<numberOfImages;i++)
		{
	  	float maxScore=0;
	  	int weightIndex=0;
			for(int w=0;w<classes;w++)
			{
	    	//variable for matrix mul
	   		float sum=0;
				for(int j=0;j<numberOfPixels;j++)
				{
					sum+= img[(i*numberOfPixels)+j]* weight[(w*numberOfPixels)+j];
				}
	      if (sum > maxScore){
	       		// Weight Having max score.
	        	maxScore = sum;
	        	weightIndex = w;
	      }
	    }
			score[i]=weightIndex;
			//std::cout<<i<<","<<weightIndex<<std::endl;
		}
 }


void linearClassifier_fxp(unsigned char *img,int *weight,int *score,int numberOfImages,int numberOfPixels,int classes){
	  for(int i=0; i<numberOfImages;i++)
		{
	  	int maxScore=0;
	  	int weightIndex=0;
			for(int w=0;w<classes;w++)
			{
	    	//variable for matrix mul
	   		int sum=0;
				for(int j=0;j<numberOfPixels;j++)
				{
					sum+= img[(i*numberOfPixels)+j]* weight[(w*numberOfPixels)+j];
				}
	      if (sum > maxScore){
	       		// Weight Having max score.
	        	maxScore = sum;
	        	weightIndex = w;
	      }
	    }
			score[i]=weightIndex;
			//std::cout<<i<<","<<weightIndex<<std::endl;
		}

 }
//Function to read CNN Conv Filter weights
bool read_cnn_weights_file_char(char *filename , std::vector<std::vector<std::vector<short>>> &CNNWeights,
  std::vector<short> &cnnbias,int FILTER_ROWS,int FILTER_COLS,int NUMBER_OF_FILTERS)
 {
      std::ifstream file(filename,std::ios::binary);

 	    int NUMBER_OF_PIXELS = ( (FILTER_ROWS*FILTER_COLS)+1) * NUMBER_OF_FILTERS ; //1 for bias
       CNNWeights.resize(NUMBER_OF_FILTERS);
       cnnbias.resize(NUMBER_OF_FILTERS);
       for(int i=0;i<NUMBER_OF_FILTERS;++i)
         {
           CNNWeights[i].resize(FILTER_ROWS);
           for(int r=0;r<FILTER_ROWS;++r)
           {
             CNNWeights[i][r].resize(FILTER_COLS);
             for(int c=0;c<FILTER_COLS;++c)
             {
               short temp=0;
               file.read((char*)&temp,sizeof(temp));
               CNNWeights[i][r][c]= (short) temp;
               }
           }
           short bias=0;
           file.read((char*)&bias,sizeof(bias));
           bias=(short)bias;
           cnnbias[i]=bias;
       }

 		return true;
 }

void convlutionLayer(std::vector<std::vector<unsigned char>> &ImageReader,std::vector<std::vector<short>> &CNNWeights,
  short cnnbias,int FILTER_ROWS,int FILTER_COLS,int NUMBER_OF_ROWS,int NUMBER_OF_COLS,
  std::vector<std::vector<long>> &ConvOutput, int CONV_LAYER_OUTPUT_ROWS, int CONV_LAYER_OUTPUT_COLS){

    //Resize the Conv Output matrix to 28*28
    ConvOutput.resize(CONV_LAYER_OUTPUT_ROWS);
    for(int i=0;i<CONV_LAYER_OUTPUT_ROWS;i++)
      ConvOutput[i].resize(CONV_LAYER_OUTPUT_COLS);

      int inX,inY=0;
      long conv=0;
      //Conv Logic
      for(int outX=0;outX<CONV_LAYER_OUTPUT_ROWS;outX++){
        for(int outY=0;outY<CONV_LAYER_OUTPUT_COLS;outY++){
          //For Input indexing
          inX = outX;
          inY = outY;
          conv=cnnbias;
          //Filter
          for(int filterX=0;filterX<FILTER_ROWS;filterX++){
            for(int filterY=0;filterY<FILTER_COLS;filterY++){
              conv+= CNNWeights[filterX][filterY] * (int)ImageReader[inX][inY];
              inY++;

            }
            inX++;
            //reset Cols
            inY=outY;

          }
          // RELU :
            conv = conv>0 ? conv :0;
          ConvOutput[outX][outY]=conv;
          conv=0;
        }
      }
}
