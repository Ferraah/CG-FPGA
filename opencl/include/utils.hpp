#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdio>
#include <cstdlib>
#include <new> // For std::align_val_t
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <map>
#include <iostream>

#include <chrono>
#include <utility>
#include <vector>
#include "CL/cl.hpp"
#include <CL/opencl.h>

#define SIZEOF_HEADER_ELEM sizeof(size_t)
#define FILE_HEADER_SIZE 2 * sizeof(size_t)

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

// Helper macro for checking OpenCL errors
#define CHECK_ERR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << msg << ": " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }


namespace utils{
    void build_source(const std::string& path, cl::Program &program, cl::Context &context, cl::Device &device);
    void load_binaries(const std::string& path, cl::Program &program, cl::Context &context, cl::Device &device);
    bool read_matrix_from_file(const char *, double *&, size_t &, size_t &);
    bool read_vector_from_file(const char * , double *& , size_t &);
    void create_vector(double * &, size_t , double );
    void create_matrix(double * &, size_t, size_t, double );
    bool read_matrix_rows(const char *, double *&, size_t , size_t , size_t &);
    bool read_matrix_dims(const char * , size_t &, size_t &);
    void print_matrix(const double * , size_t , size_t , FILE * = stdout);
}

#endif