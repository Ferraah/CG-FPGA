#ifndef FPGA_STRATEGY_HPP 
#define FPGA_STRATEGY_HPP 


#include "exception_handler.hpp"
#include "cgcore.hpp"

#include <cmath>
#include <stdio.h>
#include <cassert>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

namespace cgcore{
    
    class FPGA_CG: public CGStrategy{
        public:
            FPGA_CG();
            void run(const double * , const double * , double * , size_t , int , double ) const;

        //private: Commented for benchamrking 
            void dot(sycl::queue &q,const double *dA, const double *dB, double *dC, size_t size) const; 
            void vec_sum(sycl::queue &q,double alpha, const double *dX, double beta, double *dY, size_t size) const;
            void matrix_vector_mul(sycl::queue &q,const double *dA, const double *dB, double *dC, size_t size) const;
            void conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;

    };

}
#endif