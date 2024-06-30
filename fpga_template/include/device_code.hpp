#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP 

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

class Device{
    public:

        Device(){};
        static void dot(sycl::queue &q,const double *dA, const double *dB, double *dC, size_t size) ; 
        static void vec_sum(sycl::queue &q,double alpha, const double *dX, double beta, double *dY, size_t size) ;
        static void matrix_vector_mul(sycl::queue &q,const double *dA, const double *dB, double *dC, size_t size);
};

#endif