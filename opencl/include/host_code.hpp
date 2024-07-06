#ifndef HOST_CODE_HPP
#define HOST_CODE_HPP

#define CL_CHANNEL_1_INTELFPGA              (1<<16)
#define CL_CHANNEL_2_INTELFPGA              (2<<16)
#define CL_CHANNEL_3_INTELFPGA              (3<<16)
#define CL_CHANNEL_4_INTELFPGA              (4<<16)
#define CL_CHANNEL_5_INTELFPGA              (5<<16)
#define CL_CHANNEL_6_INTELFPGA              (6<<16)
#define CL_CHANNEL_7_INTELFPGA              (7<<16)

#define PLATFORM_ID 0


#include <stdexcept>
#include <execinfo.h>
#include <cxxabi.h>
#include <cstdlib>
#include <chrono>

//#include "exception_handler.hpp"
#include "device_code.hpp"
#include "CL/cl.hpp"
#include "utils.hpp"

void prepare(cl::Context &context, cl::CommandQueue &q, cl::Program &program, cl::Program &fblas_program );
void conjugate_gradient(cl::Device &dev, cl::Context &context, cl::CommandQueue &q, cl::Program &program, cl::Program &fblas_program, const double *A, const double *b, double *x, size_t size, int max_iters, double rel_error);

#endif