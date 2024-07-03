#include "device_code.hpp"
 

void Device::dot(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size, cl::Event &event) {
    cl_int err;
    cl::Kernel kernel(program, "DotProduct", &err);
    CHECK_ERR(err, "Failed to create Dot kernel");

    unsigned _size = size; 

    err = kernel.setArg(0, bufA);
    err |= kernel.setArg(1, bufB);
    err |= kernel.setArg(2, bufC);
    err |= kernel.setArg(3, sizeof(unsigned), &_size);
    CHECK_ERR(err, "Failed to set dot kernel arguments");

    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &event);              
    CHECK_ERR(err, "Failed to enqueue dot kernel");
}

void Device::vec_sum(double alpha, const cl::Buffer &bufX, double beta, const cl::Buffer &bufY, size_t size, cl::Event &event){
    cl_int err;
    cl::Kernel kernel(program, "VecSum", &err);
    CHECK_ERR(err, "Failed to create VecSum kernel");


    unsigned _size = size; 

    err = kernel.setArg(0, alpha);
    err |= kernel.setArg(1,bufX);
    err |= kernel.setArg(2, beta);
    err |= kernel.setArg(3, bufY);
    err |= kernel.setArg(4, sizeof(unsigned), &_size);
    CHECK_ERR(err, "Failed to set vec sum kernel arguments");

    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &event);              
    CHECK_ERR(err, "Failed to enqueue VecSum kernel");

}

void Device::matrix_vector_mul(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size, cl::Event &event){
    cl_int err;
    cl::Kernel kernel(program, "GEMV", &err);
    CHECK_ERR(err, "Failed to create GEMV kernel");

    unsigned _size = size; 

    err = kernel.setArg(0, bufA);
    err |= kernel.setArg(1, bufB);
    err |= kernel.setArg(2, bufC);
    err |= kernel.setArg(3, sizeof(unsigned), &_size);
    CHECK_ERR(err, "Failed to set gemv kernel arguments");

    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &event);              
    CHECK_ERR(err, "Failed to enqueue Gemv kernel.");
}
