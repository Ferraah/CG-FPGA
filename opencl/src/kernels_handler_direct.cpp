#include "kernels_handler.hpp"
 

void DirectKernelsHandler::dot(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size) {

    cl_int err; 
    cl::Kernel kernel(program, "ddot", &err);
    CHECK_ERR(err, "Error dot fblas kernel creation");


    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);
    kernel.setArg(3, sizeof(unsigned int), &size);

    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange); 
    CHECK_ERR(err, "Error in dot enqueue.");

    queue.finish();
    

}

void DirectKernelsHandler::vec_sum(double alpha, const cl::Buffer &bufA, double beta, const cl::Buffer &bufB, size_t size){

    cl_int err; 
    cl::Kernel kernel (program, "daxpy", &err);
    CHECK_ERR(err, "Error daxpy fblas kernel creation");


    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufB);
    kernel.setArg(3, sizeof(double), &alpha);
    kernel.setArg(4, sizeof(double), &beta);
    kernel.setArg(5, sizeof(int), &size);

   
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange); 
    CHECK_ERR(err, "Error in vec_sum enqueue.");

    queue.finish();

   
}

void DirectKernelsHandler::matrix_vector_mul(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t rows, size_t cols){

    cl_int err;
    cl::Kernel kernel(cl::Kernel(program, "dgemv", &err));
    CHECK_ERR(err, "Error gemv fblas kernel creation");

    
    const int _rows = rows;
    const int _cols = cols;
    const unsigned int one = 1;
    const double alpha = 1.0;
    const double beta = 0.0;


    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);
    kernel.setArg(3, sizeof(unsigned),&one);
    kernel.setArg(4, sizeof(int),&_rows);
    kernel.setArg(5, sizeof(int),&_cols);
    kernel.setArg(6, sizeof(double),&alpha);
    kernel.setArg(7, sizeof(double),&beta);


    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange); 
    CHECK_ERR(err, "Error in gemv enqueue.");

    queue.finish();
  
}
