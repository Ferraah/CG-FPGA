#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include <cmath>
#include <stdio.h>
#include <iostream>
    
class Sequential{
    public: 
        void run(const double * , const double * , double * , size_t , int , double );

    //private: // For debugging 
        double dot(const double * x, const double * y, size_t size) const;
        void axpby(double alpha, const double * x, double beta, double * y, size_t size) const;
        void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols) const;
        void conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;

};

#endif