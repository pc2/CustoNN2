// This file
#include "utility.h"
#include <math.h>
#include <iostream>
#include <stdio.h>

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

void fill_generate(cl_double X[], cl_double LO, cl_double HI, size_t vectorSize)
{

	//Assigns randome number from LO to HI to all locatoin of X and Y
	for (int i = 0; i < vectorSize; ++i) {
		X[i] =  LO + (cl_double)rand()/((cl_double)RAND_MAX/(HI-LO));
	}
}

bool verification (double X[], double Z[], double CalcZ, size_t vectorSize)
{
	// Print 10 Sample Data to Standard Out
	double tolerance=.01 * CalcZ;
	if ((Z[0] < CalcZ+tolerance) && (Z[0] > CalcZ-tolerance))
		printf("\n\nVERIFICATION PASSED!!!\n\n");
	else
		printf("\n\nVERIFICATION FAILED!!!\n\n");
	printf("------------------------------------\n");
	printf("FPGA Result, %f, CPU result %f\n",  Z[0], CalcZ);
	return true;
}

void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                 << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
