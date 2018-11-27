#ifndef __UTILITY_H
#define __UTILITY_H

#include "CL/cl.hpp"


#define EPSILON (1e-2f)

void print_platform_info(std::vector<cl::Platform>*);
void print_device_info(std::vector<cl::Device>*);
void fill_generate(cl_double X[], double LO, double HI, size_t vectorSize);
bool verification (double X[], double Z[], double CalcZ, size_t vectorSize);

void checkErr(cl_int err, const char * name);

#endif
