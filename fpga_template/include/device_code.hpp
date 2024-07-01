#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP 

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

class Device{
    public:

        Device(){};
        SYCL_EXTERNAL static void dot(const double *dA, const double *dB, double *dC, size_t size) ; 
        SYCL_EXTERNAL static void vec_sum(double alpha, const double *dX, double beta, double *dY, size_t size) ;
        SYCL_EXTERNAL static void matrix_vector_mul(const double *dA, const double *dB, double *dC, size_t size);
};

#endif