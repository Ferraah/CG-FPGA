#ifndef HOST_CODE_HPP
#define HOST_CODE_HPP

#include <stdexcept>
#include <execinfo.h>
#include <cxxabi.h>
#include <cstdlib>
#include <chrono>

//#include "exception_handler.hpp"
#include "device_code.hpp"



void prepare(sycl::queue &q);
void conjugate_gradient(sycl::queue &q, const double *A, const double *b, double *x, size_t size, int max_iters, double rel_error);

#endif