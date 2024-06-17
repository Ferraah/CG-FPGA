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
            void dot(const double *dA_in, const double *dB_in, double *dC_out, size_t size) const; 
            void vec_sum(const double alpha, const double *dX_in, const double beta, double *dY_in_out, size_t size) const;
            void matrix_vector_mul(const double *dA_in, const double *dB_in, double *dC_out, size_t size) const;
            void conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;

    };

}
#endif