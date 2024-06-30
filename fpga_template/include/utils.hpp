#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cassert>

#include <chrono>
#include <utility>

#include <mpi.h>

#define SIZEOF_HEADER_ELEM sizeof(size_t)
#define FILE_HEADER_SIZE 2 * sizeof(size_t)

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()


namespace utils{
    bool read_matrix_from_file(const char *, double *&, size_t &, size_t &);
    bool read_vector_from_file(const char * , double *& , size_t &);
    void create_vector(double * &, size_t , double );
    void create_matrix(double * &, size_t, size_t, double );
    bool read_matrix_rows(const char *, double *&, size_t , size_t , size_t &);
    bool read_matrix_dims(const char * , size_t &, size_t &);
    void print_matrix(const double * , size_t , size_t , FILE * = stdout);
}

#endif