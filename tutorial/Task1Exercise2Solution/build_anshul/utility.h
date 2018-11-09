#ifndef __UTILITY_H
#define __UTILITY_H

#include "CL/cl.hpp"


#define EPSILON (1e-2f)

void print_platform_info(std::vector<cl::Platform>*);
void print_device_info(std::vector<cl::Device>*);
void fill_generate(cl_float X[], cl_float Y[], cl_float Z[], float LO, float HI, size_t vectorSize);
bool verification (float X[], float Y[], float Z[], float CalcZ[], size_t vectorSize);

void checkErr(cl_int err, const char * name);

#endif
