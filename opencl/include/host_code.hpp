#ifndef HOST_CODE_HPP
#define HOST_CODE_HPP


#include <stdexcept>
#include <execinfo.h>
#include <cxxabi.h>
#include <cstdlib>
#include <chrono>

//#include "exception_handler.hpp"
#include "device_code.hpp"
#include "CL/cl.hpp"
#include "utils.hpp"

void prepare(cl::Context &context, cl::CommandQueue &q, cl::Program &program );
void prepare(cl::Context &context, cl::CommandQueue &q, cl::Program &program, int fpga_id);
void conjugate_gradient(cl::Context &context, cl::CommandQueue &q, cl::Program &program,const double *A, const double *b, double *x, size_t n, int max_iters, double rel_error, int rank, int size);

#endif